import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from utils import load_mat


# This is used to generate the diffusion view, which outputs a matrix.
def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S


datasets = ["pubmed", "Flickr"]


def gen():
    for name in datasets:
        print("loading dataset: ", name)
        data = sio.loadmat("./dataset/{}.mat".format(name))
        adj = data["Network"] if ("Network" in data) else data["A"]
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
    """Random propagation without hard-coded cuda:1.

    The original implementation used masks.cuda(cuda) and features.cuda(cuda),
    which breaks on machines with only one GPU and is inconvenient on RTX 50xx
    setups. Pass device from main.py instead.
    """
    device = _resolve_device(device=device, cuda=cuda)
    features = features.to(device)
    A = A.to(device)

    n = features.shape[0]
    drop_rates = torch.full((n,), float(dropnode_rate), dtype=torch.float32, device=device)
    masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
    features = masks * features
    return propagate(features, A, order)
