import random
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected

from card_data import (
    DEFAULT_DATA_ROOT,
    get_adj_from_mat,
    get_anomaly_label_from_mat,
    get_class_from_mat,
    get_features_from_mat,
    load_raw_mat,
)


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            shape = mx.shape
        values = mx.data
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.squeeze(np.asarray(labels_dense, dtype=np.int64))
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1, data_root=DEFAULT_DATA_ROOT):
    """Load `.mat` dataset from ~/datasets/GAD/mat by default.

    Supported keys:
      adjacency: Network / A / adj / Adj
      features:  Attributes / X / attr / features
      anomaly labels: Label / gnd / y / anomaly_label
      class labels: Class / class / labels, optional
    """
    data = load_raw_mat(dataset, data_root=data_root)
    adj = get_adj_from_mat(data)
    feat = get_features_from_mat(data)
    ano_labels = get_anomaly_label_from_mat(data)

    class_labels = get_class_from_mat(data, adj.shape[0])
    num_classes = int(np.max(class_labels)) + 1 if class_labels.size else 1
    labels = dense_to_one_hot(class_labels, max(num_classes, 1))

    if "str_anomaly_label" in data:
        str_ano_labels = np.squeeze(np.asarray(data["str_anomaly_label"]))
        attr_ano_labels = np.squeeze(np.asarray(data["attr_anomaly_label"]))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def load_mat_amazon(dataset, train_rate=0.3, val_rate=0.1, data_root=DEFAULT_DATA_ROOT):
    """Backward-compatible alias; now also loads from data_root."""
    return load_mat(dataset, train_rate=train_rate, val_rate=val_rate, data_root=data_root)


def adj_to_pyg_graph(x=None, adj=None) -> Data:
    """Convert a scipy/numpy/torch adjacency matrix into a PyG Data graph."""
    if adj is None:
        adj = x
        x = None

    if sp.issparse(adj):
        num_nodes = adj.shape[0]
        edge_index, edge_weight = from_scipy_sparse_matrix(adj.tocoo())
    elif torch.is_tensor(adj):
        num_nodes = adj.size(0)
        if adj.is_sparse:
            coalesced = adj.coalesce()
            edge_index = coalesced.indices().long().cpu()
            edge_weight = coalesced.values().float().cpu()
        else:
            sparse_adj = adj.detach().to_sparse()
            edge_index = sparse_adj.indices().long().cpu()
            edge_weight = sparse_adj.values().float().cpu()
    else:
        adj = np.asarray(adj)
        num_nodes = adj.shape[0]
        edge_index, edge_weight = from_scipy_sparse_matrix(sp.coo_matrix(adj))

    edge_index, edge_weight = to_undirected(edge_index, edge_attr=edge_weight, num_nodes=num_nodes, reduce="mean")
    if x is not None and not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index.long(), edge_weight=edge_weight, num_nodes=num_nodes)


def adj_to_dgl_graph(adj) -> Data:
    """Compatibility alias: returns PyG Data instead of a DGLGraph."""
    return adj_to_pyg_graph(None, adj)


def adj_to_dgl_graph_tensor(adj) -> Data:
    """Compatibility alias for tensor adjacency input; returns PyG Data."""
    return adj_to_pyg_graph(None, adj)


def _build_neighbor_lists(graph: Data) -> List[List[int]]:
    cached = getattr(graph, "_card_neighbor_lists", None)
    if cached is not None:
        return cached

    edge_index = graph.edge_index.detach().cpu().long()
    num_nodes = int(graph.num_nodes)
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                neighbors[u].append(v)

    for i, neigh in enumerate(neighbors):
        seen = set()
        compact = []
        for v in neigh:
            if v not in seen:
                compact.append(v)
                seen.add(v)
        neighbors[i] = compact

    graph._card_neighbor_lists = neighbors
    return neighbors


def _ordered_unique(nodes: Sequence[int]) -> List[int]:
    seen = set()
    out = []
    for node in nodes:
        node = int(node)
        if node not in seen:
            out.append(node)
            seen.add(node)
    return out


def _rwr_trace(neighbors: List[List[int]], seed: int, restart_prob: float, max_nodes_per_seed: int) -> List[int]:
    if max_nodes_per_seed <= 0:
        return [seed]
    cur = int(seed)
    trace = []
    for _ in range(max_nodes_per_seed):
        trace.append(cur)
        if random.random() < restart_prob or len(neighbors[cur]) == 0:
            cur = int(seed)
        else:
            cur = random.choice(neighbors[cur])
    return trace


def _pad_or_cut(nodes: List[int], seed: int, reduced_size: int) -> List[int]:
    if reduced_size <= 0:
        return [seed]
    if len(nodes) == 0:
        nodes = [seed]
    if len(nodes) < reduced_size:
        repeat = reduced_size // len(nodes) + 1
        nodes = (nodes * repeat)[:reduced_size]
    else:
        nodes = nodes[:reduced_size]
    nodes.append(seed)
    return nodes


def generate_rwr_subgraph(pyg_graph: Data, subgraph_size: int):
    """Generate CARD subgraphs with RWR using PyG Data instead of DGL."""
    neighbors = _build_neighbor_lists(pyg_graph)
    num_nodes = int(pyg_graph.num_nodes)
    reduced_size = subgraph_size - 1
    subgraphs = []

    for seed in range(num_nodes):
        trace = _rwr_trace(neighbors, seed, restart_prob=1.0, max_nodes_per_seed=subgraph_size * 2)
        nodes = _ordered_unique(trace)

        retry_time = 0
        while len(nodes) < reduced_size:
            trace = _rwr_trace(neighbors, seed, restart_prob=0.9, max_nodes_per_seed=subgraph_size * 4)
            nodes = _ordered_unique(trace)
            retry_time += 1
            if len(nodes) <= reduced_size and retry_time > 10:
                nodes = nodes * max(reduced_size, 1)
                break

        subgraphs.append(_pad_or_cut(nodes, seed, reduced_size))
    return subgraphs


def generate_rwr_subgraph_test(pyg_graph: Data, subgraph_size: int, adj, meanDegree):
    """Test-time RWR sampler, DGL-free PyG version."""
    neighbors = _build_neighbor_lists(pyg_graph)
    num_nodes = int(pyg_graph.num_nodes)
    reduced_size = subgraph_size - 1
    subgraphs = []

    for seed in range(num_nodes):
        trace = _rwr_trace(neighbors, seed, restart_prob=1.0, max_nodes_per_seed=int(meanDegree) * 2)
        nodes = _ordered_unique(trace)

        retry_time = 0
        while len(nodes) < reduced_size:
            trace = _rwr_trace(neighbors, seed, restart_prob=0.9, max_nodes_per_seed=subgraph_size * 4)
            nodes = _ordered_unique(trace)
            retry_time += 1
            if len(nodes) <= reduced_size and retry_time > 10:
                nodes = nodes * max(reduced_size, 1)
                break

        if len(nodes) <= reduced_size and retry_time > 10:
            subgraphs.append(_pad_or_cut(nodes, seed, reduced_size))
            continue

        degree_list = {}
        for node in nodes:
            degree = torch.sum(adj[node, :]) + torch.sum(adj[:, node])
            degree_list[int(node)] = degree

        rank_limit = len(nodes) if len(nodes) < subgraph_size * 2 else subgraph_size * 2
        rank_list = sorted(degree_list.items(), key=lambda x: x[1], reverse=True)[:rank_limit]
        choose_list = [int(item[0]) for item in rank_list] or [seed]

        tmp = []
        upper = len(choose_list) - 1 if len(nodes) < subgraph_size * 2 else min(subgraph_size * 2, len(choose_list)) - 1
        for _ in range(reduced_size):
            tmp.append(choose_list[random.randint(0, upper)])
        tmp.append(seed)
        subgraphs.append(tmp)
    return subgraphs


def generate_rwr_subgraph_v2(pyg_graph: Data, subgraph_size: int, epoch):
    """Epoch-aware RWR sampler, DGL-free PyG version."""
    restart_prob = 1.0 if epoch <= 200 else 0.8
    neighbors = _build_neighbor_lists(pyg_graph)
    num_nodes = int(pyg_graph.num_nodes)
    reduced_size = subgraph_size - 1
    subgraphs = []

    for seed in range(num_nodes):
        trace = _rwr_trace(neighbors, seed, restart_prob=restart_prob, max_nodes_per_seed=subgraph_size * 2)
        nodes = _ordered_unique(trace)

        retry_time = 0
        while len(nodes) < reduced_size:
            trace = _rwr_trace(neighbors, seed, restart_prob=0.9, max_nodes_per_seed=subgraph_size * 4)
            nodes = _ordered_unique(trace)
            retry_time += 1
            if len(nodes) <= reduced_size and retry_time > 10:
                nodes = nodes * max(reduced_size, 1)
                break

        subgraphs.append(_pad_or_cut(nodes, seed, reduced_size))
    return subgraphs
