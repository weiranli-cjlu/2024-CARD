import argparse
import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm

from aug import rand_prop
from card_data import ensure_dataset_artifacts, load_diff_csr, load_edgelist_dense
from model import Model
from utils import adj_to_pyg_graph, generate_rwr_subgraph, load_mat, normalize_adj, preprocess_features

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CARD: Community-Guided Contrastive Learning with Anomaly-Aware "
            "Reconstruction for Attributed Network Anomaly Detection"
        )
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--auc_test_rounds", type=int, default=150)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--earlystop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gama", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--data_root", type=str, default="~/datasets/GAD/mat", help="Directory containing {dataset}.mat files")
    parser.add_argument("--artifact_root", type=str, default=".", help="Where edgelist/ and diff/ are stored")
    parser.add_argument("--force_preprocess", action="store_true", help="Regenerate edgelist and diff_A even if they already exist")
    parser.add_argument("--gdc_alpha", type=float, default=0.01, help="GDC alpha for diff_A generation")
    parser.add_argument("--gdc_eps", type=float, default=0.0001, help="GDC epsilon threshold for diff_A generation")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0, cuda:1 or cpu")
    parser.add_argument("--checkpoint", type=str, default="best.pkl")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minmax_scale_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    v_min = values.min()
    v_max = values.max()
    denom = v_max - v_min
    if denom <= 1e-12:
        return np.zeros_like(values)
    return (values - v_min) / denom


def compute_metrics(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1)
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = sklearn_auc(recall, precision)
    return roc_auc, auprc


def iter_batches(num_nodes: int, batch_size: int, shuffle: bool = True):
    indices = np.arange(num_nodes)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, num_nodes, batch_size):
        yield indices[start:start + batch_size]


def gather_square_matrix(matrix: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Batch gather matrix[nodes, nodes]."""
    batch_size, subgraph_size = nodes.shape
    rows = matrix.index_select(0, nodes.reshape(-1)).reshape(batch_size, subgraph_size, -1)
    return rows.gather(2, nodes.unsqueeze(1).expand(-1, subgraph_size, -1))


def gather_node_features(features: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    batch_size, subgraph_size = nodes.shape
    return features.index_select(0, nodes.reshape(-1)).reshape(batch_size, subgraph_size, -1)


def pad_subgraph_adj(adj_batch: torch.Tensor) -> torch.Tensor:
    batch_size, subgraph_size, _ = adj_batch.shape
    padded = adj_batch.new_zeros((batch_size, subgraph_size + 1, subgraph_size + 1))
    padded[:, :subgraph_size, :subgraph_size] = adj_batch
    padded[:, -1, -1] = 1.0
    return padded


def insert_zero_before_target(feat_batch: torch.Tensor) -> torch.Tensor:
    batch_size, subgraph_size, ft_size = feat_batch.shape
    padded = feat_batch.new_zeros((batch_size, subgraph_size + 1, ft_size))
    padded[:, :-2, :] = feat_batch[:, :-1, :]
    padded[:, -1, :] = feat_batch[:, -1, :]
    return padded


def build_batch_inputs(idx, subgraphs, adj_base, diff_adj_base, modularity_base, features_base, raw_features_base, device):
    idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    nodes = subgraphs.index_select(0, idx)

    ba = pad_subgraph_adj(gather_square_matrix(adj_base, nodes))
    br = pad_subgraph_adj(gather_square_matrix(diff_adj_base, nodes))
    b_mod = pad_subgraph_adj(gather_square_matrix(modularity_base, nodes))
    bf = insert_zero_before_target(gather_node_features(features_base, nodes))
    raw = insert_zero_before_target(gather_node_features(raw_features_base, nodes))
    return bf, ba, br, raw, b_mod


def make_labels(batch_size: int, negsamp_ratio: int, device: torch.device) -> torch.Tensor:
    pos = torch.ones(batch_size, device=device)
    neg = torch.zeros(batch_size * negsamp_ratio, device=device)
    return torch.cat((pos, neg)).unsqueeze(1)


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x - y, ord=2, dim=1)


def _positive_minus_negative(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    logits = logits.reshape(-1)
    pos = logits[:batch_size]
    neg = logits[batch_size:].reshape(-1, batch_size).mean(dim=0)
    return pos - neg


def get_contrastive_scores(logits1, logits2, batch_size: int) -> np.ndarray:
    score1 = -_positive_minus_negative(logits1, batch_size)
    score2 = -_positive_minus_negative(logits2, batch_size)
    return ((score1 + score2) / 2.0).detach().cpu().numpy()


def main():
    args = parse_args()
    if args.num_epoch is None:
        args.num_epoch = 400 if args.dataset in {"ACM", "Flickr"} else 100
    if args.dataset in {"ACM", "pubmed"}:
        args.batch_size = 500

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Dataset: {args.dataset}")
    print(f"device: {device}")
    print(f"gama={args.gama}, beta={args.beta}")

    set_seed(args.seed)

    print("checking edgelist and diff_A")
    artifact_info = ensure_dataset_artifacts(
        args.dataset,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        alpha=args.gdc_alpha,
        eps=args.gdc_eps,
        force=args.force_preprocess,
    )

    normal_adj = load_edgelist_dense(artifact_info["edgelist_path"], num_nodes=artifact_info["num_nodes"])
    num_edges = max(int(artifact_info["num_edges"]), 1)
    degree = np.sum(normal_adj, axis=1)
    expected_edges = np.outer(degree, degree) / (2 * num_edges)
    modularity = normal_adj - expected_edges

    adj, features, _, _, _, _, ano_label, _, _ = load_mat(args.dataset, data_root=args.data_root)
    pyg_graph = adj_to_pyg_graph(None, adj)

    diff_adj = load_diff_csr(artifact_info["diff_path"])
    diff_adj = (diff_adj + sp.eye(diff_adj.shape[0])).todense()

    raw_feature = features.todense()
    features, _ = preprocess_features(features)
    nb_nodes, ft_size = features.shape

    c_features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    c_adj = torch.as_tensor(adj.todense(), dtype=torch.float32, device=device)
    c_features = rand_prop(features=c_features, dropnode_rate=0.5, A=c_adj, order=5, device=device).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    raw_feature = torch.as_tensor(np.asarray(raw_feature), dtype=torch.float32, device=device)
    adj = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)
    diff_adj = torch.as_tensor(np.asarray(diff_adj), dtype=torch.float32, device=device)
    modularity = torch.as_tensor(np.asarray(modularity), dtype=torch.float32, device=device)
    c_features = c_features.to(device)

    alpha = 0.3 if args.dataset in {"cora", "citeseer"} else 0.1
    model = Model(
        ft_size,
        args.embedding_dim,
        "prelu",
        args.negsamp_ratio,
        args.readout,
        args.dropout,
        args.subgraph_size,
        adj.shape[0],
        alpha=alpha,
    ).to(device)

    print("the running gama is %f, fpbal is %f" % (args.gama, args.beta))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
    )
    mse_loss = nn.MSELoss(reduction="mean")

    cnt_wait = 0
    best_loss = float("inf")
    best_epoch = 0
    best_auc = 0.0
    checkpoint = Path(args.checkpoint)

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.0
            num_batches = 0
            subgraphs = torch.as_tensor(generate_rwr_subgraph(pyg_graph, args.subgraph_size), dtype=torch.long, device=device)

            for batch_no, idx in enumerate(iter_batches(nb_nodes, args.batch_size, shuffle=True)):
                is_final_batch = batch_no == (nb_nodes - 1) // args.batch_size
                cur_batch_size = len(idx)
                lbl = make_labels(cur_batch_size, args.negsamp_ratio, device)
                bf, ba, br, raw, b_mod = build_batch_inputs(
                    idx, subgraphs, adj, diff_adj, modularity, features, raw_feature, device
                )

                optimiser.zero_grad(set_to_none=True)
                now1, logits1, kl_1 = model(bf, ba, raw, b_mod)
                now2, logits2, kl_2 = model(bf, br, raw, b_mod)

                kl = 0.5 * (kl_2 + kl_1)
                loss_re = 0.5 * (mse_loss(now1, raw[:, -1, :]) + mse_loss(now2, raw[:, -1, :]))
                loss_bce = 0.5 * (b_xent(logits1, lbl) + b_xent(logits2, lbl))

                h_1 = F.normalize(logits1[:cur_batch_size, :], dim=1, p=2)
                h_2 = F.normalize(logits2[:cur_batch_size, :], dim=1, p=2)
                coloss = 2 - 2 * (h_1 * h_2).sum(dim=-1).mean()

                local_loss = torch.mean(loss_bce) + coloss + args.gama * loss_re
                if is_final_batch:
                    c_now = model.global_reconstruct(c_features.unsqueeze(0), adj.unsqueeze(0))
                    loss_global_re = pairwise_l2(c_now[0], raw_feature).mean() * (args.batch_size / adj.shape[0])
                    loss = (1 - args.beta) * local_loss + args.beta * loss_global_re + 0.5 * kl
                else:
                    loss = (1 - args.beta) * local_loss + 0.5 * kl

                loss.backward()
                optimiser.step()

                total_loss += float(loss.detach().cpu())
                num_batches += 1

            mean_loss = total_loss / max(num_batches, 1)

            if args.earlystop:
                model.eval()
                with torch.no_grad():
                    c_now = model.global_reconstruct(c_features.unsqueeze(0), adj.unsqueeze(0))
                    global_score = minmax_scale_np(pairwise_l2(c_now[0], raw_feature).cpu().numpy())
                    val_auc, _ = compute_metrics(ano_label, global_score)
                if val_auc > best_auc:
                    best_auc = val_auc
                else:
                    cnt_wait += 1
                    if cnt_wait > 200:
                        pbar.update(1)
                        break

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_epoch = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=f"{mean_loss:.4f}", best_epoch=best_epoch)
            pbar.update(1)

    print("testing_" + args.dataset)
    print("Loading {}th epoch from the training".format(best_epoch))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)
    multi_round_ano_score_global = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)

    with torch.no_grad(), tqdm(total=args.auc_test_rounds, desc="EVALUATION CARD") as pbar_test:
        for round_idx in range(args.auc_test_rounds):
            subgraphs = torch.as_tensor(generate_rwr_subgraph(pyg_graph, args.subgraph_size), dtype=torch.long, device=device)

            c_now = model.global_reconstruct(c_features.unsqueeze(0), adj.unsqueeze(0))
            global_score = pairwise_l2(c_now[0], raw_feature).cpu().numpy()
            multi_round_ano_score_global[round_idx, :] = minmax_scale_np(global_score)

            for idx in iter_batches(nb_nodes, args.batch_size, shuffle=True):
                cur_batch_size = len(idx)
                bf, ba, br, raw, b_mod = build_batch_inputs(
                    idx, subgraphs, adj, diff_adj, modularity, features, raw_feature, device
                )

                now1, logits1, _ = model(bf, ba, raw, b_mod)
                now2, logits2, _ = model(bf, br, raw, b_mod)
                logits1 = torch.sigmoid(torch.squeeze(logits1))
                logits2 = torch.sigmoid(torch.squeeze(logits2))

                score_co = minmax_scale_np(get_contrastive_scores(logits1, logits2, cur_batch_size))
                score_re = 0.5 * (pairwise_l2(now1, raw[:, -1, :]) + pairwise_l2(now2, raw[:, -1, :]))
                score_re = minmax_scale_np(score_re.cpu().numpy())
                multi_round_ano_score[round_idx, idx] = score_co + args.gama * score_re

            pbar_test.update(1)

    ano_score_final = (
        (1 - args.beta) * np.mean(multi_round_ano_score, axis=0)
        + args.beta * np.mean(multi_round_ano_score_global, axis=0)
    )
    roc_auc, auprc = compute_metrics(ano_label, ano_score_final)
    print("the auc is ", roc_auc)
    print("the auprc is ", auprc)


if __name__ == "__main__":
    main()
