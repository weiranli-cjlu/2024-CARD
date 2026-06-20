import argparse
import copy
import csv
import gc
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

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


@dataclass
class DatasetTensors:
    """All data needed by CARD after preprocessing."""

    name: str
    labels: np.ndarray
    pyg_graph: object
    num_nodes: int
    feature_dim: int
    adj: torch.Tensor
    diff_adj: torch.Tensor
    modularity: torch.Tensor
    features: torch.Tensor
    raw_features: torch.Tensor
    contrast_features: torch.Tensor


@dataclass
class TrialResult:
    seed: int
    auc: float
    auprc: float


def train_args():
    parser = argparse.ArgumentParser(
        description="CARD training/evaluation for one dataset at a time."
    )
    parser.add_argument("--dataset", type=str, default="cora", help="Single dataset name, e.g. cora.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2, help="Base seed. Trial i uses seed + i * seed_step.")
    parser.add_argument("--seed_step", type=int, default=1, help="Seed increment between repeated trials.")
    parser.add_argument("--trials", type=int, default=10, help="Number of repeated runs on the same dataset.")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--auc_test_rounds", type=int, default=150)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--earlystop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=200, help="Stop if training loss does not improve for N epochs.")
    parser.add_argument("--gama", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--data_root", type=str, default="~/datasets/GAD/mat", help="Directory containing {dataset}.mat files.")
    parser.add_argument("--artifact_root", type=str, default=".", help="Where edgelist/ and diff/ are stored.")
    parser.add_argument("--force_preprocess", action="store_true", help="Regenerate edgelist and diff_A even if they exist.")
    parser.add_argument("--gdc_alpha", type=float, default=0.01, help="GDC alpha for diff_A generation.")
    parser.add_argument("--gdc_eps", type=float, default=0.0001, help="GDC epsilon threshold for diff_A generation.")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0, cuda:1 or cpu.")
    parser.add_argument("--result_csv", type=str, default="results/card_multitrial.csv", help="CSV file used to append trial summary.")
    parser.add_argument("--no_save_csv", action="store_true", help="Do not append summary to CSV.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if any(sep in args.dataset for sep in [",", " ", "\t"]):
        raise ValueError("--dataset only accepts one dataset name. Run the script again for another dataset.")


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


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    """Keep the original CARD defaults, but apply them in one obvious place."""
    if args.num_epoch is None:
        args.num_epoch = 400 if args.dataset in {"ACM", "Flickr"} else 100
    if args.dataset in {"ACM", "pubmed"}:
        args.batch_size = 500


def minmax_scale_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    value_range = values.max() - values.min()
    if value_range <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Return AUROC and AUPRC as decimals in [0, 1]."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1)
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = sklearn_auc(recall, precision)
    return roc_auc, auprc


def format_metric(values: Iterable[float]) -> str:
    """Format decimal metric values as percentage mean±std(max)."""
    arr = np.asarray(list(values), dtype=np.float64) * 100.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0
    best = float(np.max(arr))
    return f"{mean:.2f}±{std:.2f}({best:.2f})"


def append_result_csv(path: str, row: dict) -> None:
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["datetime", "trial", "dataset", "auc", "auprc"]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def iter_batches(num_nodes: int, batch_size: int, shuffle: bool = True):
    indices = np.arange(num_nodes)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, num_nodes, batch_size):
        yield indices[start:start + batch_size]


def gather_square_matrix(matrix: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
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
    batch_size, subgraph_size, feature_dim = feat_batch.shape
    padded = feat_batch.new_zeros((batch_size, subgraph_size + 1, feature_dim))
    padded[:, :-2, :] = feat_batch[:, :-1, :]
    padded[:, -1, :] = feat_batch[:, -1, :]
    return padded


def build_batch_inputs(
    idx,
    subgraphs: torch.Tensor,
    data: DatasetTensors,
    device: torch.device,
):
    idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    nodes = subgraphs.index_select(0, idx)
    batch_adj = pad_subgraph_adj(gather_square_matrix(data.adj, nodes))
    batch_diff_adj = pad_subgraph_adj(gather_square_matrix(data.diff_adj, nodes))
    batch_modularity = pad_subgraph_adj(gather_square_matrix(data.modularity, nodes))
    batch_features = insert_zero_before_target(gather_node_features(data.features, nodes))
    batch_raw_features = insert_zero_before_target(gather_node_features(data.raw_features, nodes))
    return batch_features, batch_adj, batch_diff_adj, batch_raw_features, batch_modularity


def make_labels(batch_size: int, negsamp_ratio: int, device: torch.device) -> torch.Tensor:
    pos = torch.ones(batch_size, device=device)
    neg = torch.zeros(batch_size * negsamp_ratio, device=device)
    return torch.cat((pos, neg)).unsqueeze(1)


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x - y, ord=2, dim=1)


def positive_minus_negative(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    logits = logits.reshape(-1)
    pos = logits[:batch_size]
    neg = logits[batch_size:].reshape(-1, batch_size).mean(dim=0)
    return pos - neg


def get_contrastive_scores(logits1: torch.Tensor, logits2: torch.Tensor, batch_size: int) -> np.ndarray:
    score1 = -positive_minus_negative(logits1, batch_size)
    score2 = -positive_minus_negative(logits2, batch_size)
    return ((score1 + score2) / 2.0).detach().cpu().numpy()


def prepare_dataset(args: argparse.Namespace, device: torch.device) -> DatasetTensors:
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

    adj, features, _, _, _, _, labels, _, _ = load_mat(args.dataset, data_root=args.data_root)
    pyg_graph = adj_to_pyg_graph(None, adj)

    diff_adj = load_diff_csr(artifact_info["diff_path"])
    diff_adj = (diff_adj + sp.eye(diff_adj.shape[0], format="csr")).todense()

    raw_features = features.todense()
    norm_features, _ = preprocess_features(features)
    num_nodes, feature_dim = norm_features.shape

    contrast_features = torch.as_tensor(np.asarray(norm_features), dtype=torch.float32)
    dense_adj_for_prop = torch.as_tensor(adj.todense(), dtype=torch.float32, device=device)
    contrast_features = rand_prop(
        features=contrast_features,
        dropnode_rate=0.5,
        A=dense_adj_for_prop,
        order=5,
        device=device,
    ).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    norm_adj = normalize_adj(adj)
    norm_adj = (norm_adj + sp.eye(norm_adj.shape[0], format="csr")).todense()

    return DatasetTensors(
        name=args.dataset,
        labels=labels,
        pyg_graph=pyg_graph,
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        adj=torch.as_tensor(np.asarray(norm_adj), dtype=torch.float32, device=device),
        diff_adj=torch.as_tensor(np.asarray(diff_adj), dtype=torch.float32, device=device),
        modularity=torch.as_tensor(np.asarray(modularity), dtype=torch.float32, device=device),
        features=torch.as_tensor(np.asarray(norm_features), dtype=torch.float32, device=device),
        raw_features=torch.as_tensor(np.asarray(raw_features), dtype=torch.float32, device=device),
        contrast_features=contrast_features.to(device),
    )


def build_model(args: argparse.Namespace, data: DatasetTensors, device: torch.device) -> Model:
    alpha = 0.3 if args.dataset in {"cora", "citeseer"} else 0.1
    return Model(
        data.feature_dim,
        args.embedding_dim,
        "prelu",
        args.negsamp_ratio,
        args.readout,
        args.dropout,
        args.subgraph_size,
        data.num_nodes,
        alpha=alpha,
    ).to(device)


def train_one_epoch(
    model: Model,
    data: DatasetTensors,
    args: argparse.Namespace,
    optimiser: torch.optim.Optimizer,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    subgraphs = torch.as_tensor(
        generate_rwr_subgraph(data.pyg_graph, args.subgraph_size),
        dtype=torch.long,
        device=device,
    )

    for batch_no, idx in enumerate(iter_batches(data.num_nodes, args.batch_size, shuffle=True)):
        is_final_batch = batch_no == (data.num_nodes - 1) // args.batch_size
        cur_batch_size = len(idx)
        labels = make_labels(cur_batch_size, args.negsamp_ratio, device)
        batch_features, batch_adj, batch_diff_adj, batch_raw, batch_modularity = build_batch_inputs(
            idx, subgraphs, data, device
        )

        optimiser.zero_grad(set_to_none=True)
        recon1, logits1, kl1 = model(batch_features, batch_adj, batch_raw, batch_modularity)
        recon2, logits2, kl2 = model(batch_features, batch_diff_adj, batch_raw, batch_modularity)
        kl = 0.5 * (kl1 + kl2)

        recon_loss = 0.5 * (mse_loss(recon1, batch_raw[:, -1, :]) + mse_loss(recon2, batch_raw[:, -1, :]))
        contrastive_loss = 0.5 * (bce_loss(logits1, labels) + bce_loss(logits2, labels))
        h1 = F.normalize(logits1[:cur_batch_size, :], dim=1, p=2)
        h2 = F.normalize(logits2[:cur_batch_size, :], dim=1, p=2)
        consistency_loss = 2 - 2 * (h1 * h2).sum(dim=-1).mean()
        local_loss = torch.mean(contrastive_loss) + consistency_loss + args.gama * recon_loss

        if is_final_batch:
            global_recon = model.global_reconstruct(data.contrast_features.unsqueeze(0), data.adj.unsqueeze(0))
            global_loss = pairwise_l2(global_recon[0], data.raw_features).mean() * (args.batch_size / data.num_nodes)
            loss = (1 - args.beta) * local_loss + args.beta * global_loss + 0.5 * kl
        else:
            loss = (1 - args.beta) * local_loss + 0.5 * kl

        loss.backward()
        optimiser.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(
    model: Model,
    data: DatasetTensors,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print("the running gama is %f, fpbal is %f" % (args.gama, args.beta))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
    )
    mse_loss = nn.MSELoss(reduction="mean")

    best_loss = float("inf")
    best_epoch = -1
    wait = 0

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            mean_loss = train_one_epoch(model, data, args, optimiser, bce_loss, mse_loss, device)

            if mean_loss < best_loss - 1e-12:
                best_loss = mean_loss
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            pbar.set_postfix(loss=f"{mean_loss:.4f}", best_epoch=best_epoch, wait=wait)
            pbar.update(1)

            if args.earlystop and wait >= args.patience:
                print(f"Early stop at epoch {epoch}; best training loss was at epoch {best_epoch}.")
                break


def evaluate_model(
    model: Model,
    data: DatasetTensors,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float]:
    print("testing_" + args.dataset)
    model.eval()

    local_scores = np.zeros((args.auc_test_rounds, data.num_nodes), dtype=np.float32)
    global_scores = np.zeros((args.auc_test_rounds, data.num_nodes), dtype=np.float32)

    with torch.no_grad(), tqdm(total=args.auc_test_rounds, desc="EVALUATION CARD") as pbar_test:
        for round_idx in range(args.auc_test_rounds):
            subgraphs = torch.as_tensor(
                generate_rwr_subgraph(data.pyg_graph, args.subgraph_size),
                dtype=torch.long,
                device=device,
            )
            global_recon = model.global_reconstruct(data.contrast_features.unsqueeze(0), data.adj.unsqueeze(0))
            global_score = pairwise_l2(global_recon[0], data.raw_features).cpu().numpy()
            global_scores[round_idx, :] = minmax_scale_np(global_score)

            for idx in iter_batches(data.num_nodes, args.batch_size, shuffle=False):
                cur_batch_size = len(idx)
                batch_features, batch_adj, batch_diff_adj, batch_raw, batch_modularity = build_batch_inputs(
                    idx, subgraphs, data, device
                )
                recon1, logits1, _ = model(batch_features, batch_adj, batch_raw, batch_modularity)
                recon2, logits2, _ = model(batch_features, batch_diff_adj, batch_raw, batch_modularity)
                logits1 = torch.sigmoid(torch.squeeze(logits1))
                logits2 = torch.sigmoid(torch.squeeze(logits2))

                score_co = minmax_scale_np(get_contrastive_scores(logits1, logits2, cur_batch_size))
                score_re = 0.5 * (pairwise_l2(recon1, batch_raw[:, -1, :]) + pairwise_l2(recon2, batch_raw[:, -1, :]))
                score_re = minmax_scale_np(score_re.cpu().numpy())
                local_scores[round_idx, idx] = score_co + args.gama * score_re
            pbar_test.update(1)

    final_score = (1 - args.beta) * np.mean(local_scores, axis=0) + args.beta * np.mean(global_scores, axis=0)
    roc_auc, auprc = compute_metrics(data.labels, final_score)
    print("the auc is ", roc_auc)
    print("the auprc is ", auprc)
    return roc_auc, auprc


def run_single_trial(args: argparse.Namespace) -> TrialResult:
    apply_dataset_defaults(args)
    device = get_device(args.device)

    print(f"Dataset: {args.dataset}")
    print(f"device: {device}")
    print(f"seed={args.seed}, gama={args.gama}, beta={args.beta}")

    set_seed(args.seed)
    data = prepare_dataset(args, device)
    model = build_model(args, data, device)
    train_model(model, data, args, device)
    auc_value, auprc_value = evaluate_model(model, data, args, device)
    return TrialResult(seed=args.seed, auc=auc_value, auprc=auprc_value)


def run_all_trials(args: argparse.Namespace) -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial_idx in range(args.trials):
        trial_args = copy.copy(args)
        trial_args.seed = args.seed + trial_idx * args.seed_step
        print(f"\n========== CARD trial {trial_idx + 1}/{args.trials}, seed={trial_args.seed} ==========")

        result = run_single_trial(trial_args)
        results.append(result)
        print(f"[trial {trial_idx + 1}/{args.trials}] auc={result.auc:.6f}, auprc={result.auprc:.6f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def save_summary(args: argparse.Namespace, results: list[TrialResult]) -> None:
    auc_values = [result.auc for result in results]
    auprc_values = [result.auprc for result in results]
    summary = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trial": len(results),
        "dataset": args.dataset,
        "auc": format_metric(auc_values),
        "auprc": format_metric(auprc_values),
    }

    print("\n========== Trial summary ==========")
    print(summary)

    if not args.no_save_csv:
        append_result_csv(args.result_csv, summary)
        print(f"Saved CSV to: {Path(args.result_csv).expanduser()}")


def main() -> None:
    args = train_args().parse_args()
    validate_args(args)
    results = run_all_trials(args)
    save_summary(args, results)


if __name__ == "__main__":
    main()
