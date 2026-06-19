"""Dataset and artifact helpers for CARD.

This module makes CARD read `.mat` files from `~/datasets/GAD/mat` by default
and automatically creates the two runtime artifacts used by the original code:

  * `edgelist/{dataset}.edgelist`
  * `diff/diff_A_{dataset}.npy`

The functions are intentionally independent from DGL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


DEFAULT_DATA_ROOT = "~/datasets/GAD/mat"


ADJ_KEYS = ("Network", "A", "adj", "Adj", "network")
FEAT_KEYS = ("Attributes", "X", "attr", "features", "Features")
ANOMALY_LABEL_KEYS = ("Label", "gnd", "y", "Y", "ano_label", "anomaly_label")
CLASS_KEYS = ("Class", "class", "labels", "Labels")


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def get_mat_path(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    root = expand_path(data_root)
    filename = dataset if dataset.endswith(".mat") else f"{dataset}.mat"
    return root / filename


def _first_existing_key(data: Dict, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def load_raw_mat(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Dict:
    mat_path = get_mat_path(dataset, data_root)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Cannot find dataset file: {mat_path}\n"
            f"Expected .mat file under: {expand_path(data_root)}"
        )
    return sio.loadmat(mat_path)


def get_adj_from_mat(data: Dict) -> sp.csr_matrix:
    key = _first_existing_key(data, ADJ_KEYS)
    if key is None:
        raise KeyError(f"No adjacency field found. Tried: {ADJ_KEYS}")
    return sp.csr_matrix(data[key])


def get_features_from_mat(data: Dict) -> sp.lil_matrix:
    key = _first_existing_key(data, FEAT_KEYS)
    if key is None:
        raise KeyError(f"No feature field found. Tried: {FEAT_KEYS}")
    return sp.lil_matrix(data[key])


def get_anomaly_label_from_mat(data: Dict) -> np.ndarray:
    key = _first_existing_key(data, ANOMALY_LABEL_KEYS)
    if key is None:
        raise KeyError(f"No anomaly label field found. Tried: {ANOMALY_LABEL_KEYS}")
    return np.squeeze(np.asarray(data[key]))


def get_class_from_mat(data: Dict, num_nodes: int) -> np.ndarray:
    key = _first_existing_key(data, CLASS_KEYS)
    if key is None:
        return np.zeros(num_nodes, dtype=np.int64)
    labels = np.squeeze(np.asarray(data[key], dtype=np.int64))
    # Some CARD datasets store class labels from 1..C. Convert those to 0..C-1.
    if labels.size > 0 and labels.min() == 1:
        labels = labels - 1
    return labels.astype(np.int64)


def make_undirected_binary(adj: sp.spmatrix) -> sp.csr_matrix:
    """Return an undirected, unweighted adjacency matrix with zero diagonal."""
    adj = sp.csr_matrix(adj)
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def compute_edge_count(adj: sp.spmatrix) -> int:
    """Number of undirected edges in a symmetric adjacency matrix."""
    adj = make_undirected_binary(adj)
    return int(sp.triu(adj, k=1).nnz)


def save_edgelist_from_adj(adj: sp.spmatrix, out_path: str | Path) -> int:
    """Save an undirected weighted edge list as `src dst weight`.

    The file preserves original integer node ids. Isolated nodes are not listed,
    so `load_edgelist_dense(..., num_nodes=...)` should be used when reading.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adj = make_undirected_binary(adj)
    upper = sp.triu(adj, k=1).tocoo()
    num_edges = int(upper.nnz)

    if num_edges == 0:
        out_path.write_text("", encoding="utf-8")
        return 0

    edges = np.column_stack(
        [upper.row.astype(np.int64), upper.col.astype(np.int64), np.ones(num_edges, dtype=np.float32)]
    )
    np.savetxt(out_path, edges, fmt=["%d", "%d", "%.1f"])
    return num_edges


def load_edgelist_dense(path: str | Path, num_nodes: Optional[int] = None) -> np.ndarray:
    """Load CARD edge list into a dense adjacency matrix.

    This replaces the original `nx.read_weighted_edgelist` path and keeps the
    number/order of nodes consistent with the `.mat` file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if num_nodes is None:
        # Infer from max id if possible.
        if path.stat().st_size == 0:
            raise ValueError("num_nodes is required for an empty edgelist")
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)
        num_nodes = int(arr[:, :2].max()) + 1

    A = np.zeros((int(num_nodes), int(num_nodes)), dtype=np.float32)
    if path.stat().st_size == 0:
        return A

    arr = np.loadtxt(path)
    arr = np.atleast_2d(arr)
    src = arr[:, 0].astype(np.int64)
    dst = arr[:, 1].astype(np.int64)
    if arr.shape[1] >= 3:
        weight = arr[:, 2].astype(np.float32)
    else:
        weight = np.ones(src.shape[0], dtype=np.float32)
    A[src, dst] = weight
    A[dst, src] = weight
    return A


def gdc(A: sp.csr_matrix, alpha: float = 0.01, eps: float = 0.0001) -> sp.csr_matrix:
    """Generate the diffusion graph used by CARD.

    This is the same logic as CARD's `aug.py:gdc`, kept here so preprocessing
    does not depend on importing the training augmentation module.
    """
    A = sp.csr_matrix(A)
    N = A.shape[0]
    A_loop = sp.eye(N, format="csr") + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1.0 / np.sqrt(D_loop_vec)
    D_loop_vec_invsqrt[np.isinf(D_loop_vec_invsqrt)] = 0.0
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N, format="csr") - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    D_tilde_vec[D_tilde_vec == 0] = 1.0
    T_S = S_tilde / D_tilde_vec
    return sp.csr_matrix(T_S)


def save_diff(adj: sp.spmatrix, out_path: str | Path, alpha: float = 0.01, eps: float = 0.0001) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff = gdc(sp.csr_matrix(adj), alpha=alpha, eps=eps)
    # Save as a scipy sparse object. Use load_diff_csr() below to read it back.
    np.save(out_path, diff)


def load_diff_csr(path: str | Path) -> sp.csr_matrix:
    """Load diff_A saved either as sparse-object npy or dense npy."""
    obj = np.load(Path(path), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        item = obj.item()
        if sp.issparse(item):
            return sp.csr_matrix(item)
        return sp.csr_matrix(item)
    return sp.csr_matrix(obj)


def ensure_dataset_artifacts(
    dataset: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    artifact_root: str | Path = ".",
    alpha: float = 0.01,
    eps: float = 0.0001,
    force: bool = False,
) -> Dict[str, object]:
    """Ensure edgelist and diffusion matrix exist before CARD training.

    Returns paths and graph statistics used by `main.py`.
    """
    data = load_raw_mat(dataset, data_root=data_root)
    raw_adj = get_adj_from_mat(data)
    adj = make_undirected_binary(raw_adj)
    num_nodes = int(adj.shape[0])

    root = expand_path(artifact_root)
    edgelist_path = root / "edgelist" / f"{dataset}.edgelist"
    diff_path = root / "diff" / f"diff_A_{dataset}.npy"

    if force or not edgelist_path.exists():
        num_edges = save_edgelist_from_adj(adj, edgelist_path)
        print(f"[auto] generated edgelist: {edgelist_path}")
    else:
        num_edges = compute_edge_count(adj)
        print(f"[auto] found edgelist: {edgelist_path}")

    if force or not diff_path.exists():
        print(f"[auto] generating diff_A: {diff_path}")
        print("[auto] note: GDC uses a matrix inverse and may be slow/memory-heavy on large graphs")
        save_diff(adj, diff_path, alpha=alpha, eps=eps)
    else:
        print(f"[auto] found diff_A: {diff_path}")

    return {
        "mat_path": get_mat_path(dataset, data_root),
        "edgelist_path": str(edgelist_path),
        "diff_path": str(diff_path),
        "num_nodes": num_nodes,
        "num_edges": int(num_edges),
    }
