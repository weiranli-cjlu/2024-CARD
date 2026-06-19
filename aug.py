import numpy as np
import scipy.sparse as sp
import torch

from card_data import DEFAULT_DATA_ROOT, get_adj_from_mat, load_raw_mat


# This is used to generate the diffusion view, which outputs a matrix.
def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N, format="csr") + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_vec_invsqrt[np.isinf(D_loop_vec_invsqrt)] = 0.0
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N, format="csr") - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    D_tilde_vec[D_tilde_vec == 0] = 1.0
    T_S = S_tilde / D_tilde_vec
    return T_S


datasets = ["pubmed", "Flickr"]


def gen(data_root=DEFAULT_DATA_ROOT):
    for name in datasets:
        print("loading dataset: ", name)
        data = load_raw_mat(name, data_root=data_root)
        adj = get_adj_from_mat(data)
        print("generating dataset", name)
        diff = gdc(adj, alpha=0.01, eps=0.0001)
        np.save("./diff/diff_A_" + name, diff)
        print("generating " + name + " finished")


def propagate(feature, A, order):
    x = feature
    y = feature
    for _ in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order + 1.0).detach_()


def _resolve_device(device=None, cuda=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        if cuda is None:
            return torch.device("cuda:0")
        return torch.device(f"cuda:{cuda}")
    return torch.device("cpu")


def rand_prop(features, dropnode_rate, A, order, cuda=None, device=None):
    """Random propagation without hard-coded cuda:1."""
    device = _resolve_device(device=device, cuda=cuda)
    features = features.to(device)
    A = A.to(device)

    n = features.shape[0]
    drop_rates = torch.full((n,), float(dropnode_rate), dtype=torch.float32, device=device)
    masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
    features = masks * features
    return propagate(features, A, order)
