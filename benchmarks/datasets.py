"""Benchmark datasets: vectors to index, queries, and their exact neighbours.

A dataset is built by a loader function registered in DATASETS. Every loader
returns the same Dataset shape, so run.py never needs to know where the data
came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Dataset:
    name: str
    metric: str  # "l2" | "ip" | "cosine", same names as annex.FlatIndex
    train: np.ndarray  # (n, dim) float32, the vectors that are indexed
    queries: np.ndarray  # (nq, dim) float32
    ground_truth: np.ndarray  # (nq, k_gt) uint64, labels (= train row numbers) nearest first

    @property
    def dim(self) -> int:
        return self.train.shape[1]


def _distances(train: np.ndarray, queries: np.ndarray, metric: str) -> np.ndarray:
    """(nq, n) distances with annex's conventions: smaller is more similar."""
    if metric == "l2":
        # ||q - t||^2 = ||q||^2 - 2 q.t + ||t||^2. One matrix product instead of
        # an (nq, n, dim) broadcast, which would need far more memory.
        q2 = (queries**2).sum(axis=1)[:, None]
        t2 = (train**2).sum(axis=1)[None, :]
        return q2 - 2.0 * (queries @ train.T) + t2
    if metric == "ip":
        return 1.0 - queries @ train.T
    if metric == "cosine":
        t = train / np.linalg.norm(train, axis=1, keepdims=True)
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        return 1.0 - q @ t.T
    raise ValueError(f"unknown metric {metric!r}")


def exact_ground_truth(
    train: np.ndarray, queries: np.ndarray, k: int, metric: str, chunk: int = 256
) -> np.ndarray:
    """Exact k nearest neighbours of each query, by brute force in NumPy.

    Queries are processed `chunk` at a time so peak memory is chunk * n
    distances rather than nq * n. Distances are computed in float64 so that
    rounding cannot reorder near-ties differently from one chunk to the next;
    the engines under test compute in float32, so recall can in principle drop
    on a genuine near-tie, which is a property of the data, not a bug.
    """
    k = min(k, train.shape[0])
    out = np.empty((queries.shape[0], k), dtype=np.uint64)
    if k == 0:
        return out
    train64 = train.astype(np.float64)
    for start in range(0, queries.shape[0], chunk):
        q = queries[start : start + chunk].astype(np.float64)
        d = _distances(train64, q, metric)
        # argpartition finds the k smallest in O(n); only those k are sorted.
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        # Sort by distance, then by row number (= label), the same order as
        # annex's Neighbor::operator<. lexsort's last key is the primary one.
        order = np.lexsort((part, part_d), axis=1)
        out[start : start + q.shape[0]] = np.take_along_axis(part, order, axis=1)
    return out


def synthetic(n: int, dim: int, queries: int, k: int, metric: str, seed: int) -> Dataset:
    """Seeded i.i.d. Gaussian vectors.

    Useful as a smoke test and for quick comparisons, but easy for ANN indexes
    compared with real embeddings, which have structure (clusters, low
    intrinsic dimension). Real datasets arrive in M2.
    """
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, dim), dtype=np.float32)
    query_vectors = rng.standard_normal((queries, dim), dtype=np.float32)
    gt = exact_ground_truth(train, query_vectors, k, metric)
    return Dataset(
        name="synthetic", metric=metric, train=train, queries=query_vectors, ground_truth=gt
    )


# Name -> loader. Every loader takes the same keyword arguments; a loader for a
# fixed dataset may ignore the ones that do not apply to it (e.g. n, dim).
DATASETS: dict[str, Callable[..., Dataset]] = {
    "synthetic": synthetic,
    # M2: ann-benchmarks HDF5 loaders go here ("sift-128-euclidean",
    # "glove-100-angular"). They download into benchmarks/data/ and use the
    # file's own queries and ground truth instead of exact_ground_truth().
}


def load(name: str, **kwargs) -> Dataset:
    try:
        loader = DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from None
    return loader(**kwargs)
