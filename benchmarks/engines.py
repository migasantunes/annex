"""Search engines behind one small interface, so run.py can time any of them.

To add an engine, subclass Engine and register it in ENGINES. Imports of
optional libraries (hnswlib, faiss) belong inside build() or __init__, so the
harness still runs when they are not installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Engine(ABC):
    """One index implementation under test.

    Lifecycle: build() once, then any number of search() calls. Engines are
    labelled by train row number, so results compare directly with
    Dataset.ground_truth.
    """

    name: str

    @abstractmethod
    def build(self, train: np.ndarray, metric: str) -> None:
        """Index `train` ((n, dim) float32) for the given metric."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """(nq, k) labels of the nearest neighbours, nearest first."""

    def set_query_param(self, name: str, value: object) -> None:
        """Change a search-time parameter without rebuilding.

        M2 sweeps ef_search through this to draw a recall/QPS curve. Exact
        engines have no such knob, so the default refuses everything.
        """
        raise ValueError(f"engine {self.name!r} has no query parameter {name!r}")


class AnnexFlat(Engine):
    """annex.FlatIndex: exact search in C++."""

    name = "annex-flat"

    def build(self, train: np.ndarray, metric: str) -> None:
        import annex

        self._index = annex.FlatIndex(dim=train.shape[1], metric=metric)
        self._index.add(train)  # default labels 0..n-1 == row numbers

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        labels, _ = self._index.search(queries, k)
        return labels


class NumpyFlat(Engine):
    """Exact search in NumPy, one matrix product per call.

    Two jobs: it proves the Engine interface is not shaped around annex, and it
    is a baseline that says whether the C++ index is worth having at all.
    Uses float32 throughout, as a user of NumPy would.
    """

    name = "numpy-flat"

    def build(self, train: np.ndarray, metric: str) -> None:
        if metric not in ("l2", "ip", "cosine"):
            raise ValueError(f"unknown metric {metric!r}")
        self._metric = metric
        self._train = np.ascontiguousarray(train, dtype=np.float32)
        if metric == "cosine":
            # Normalise once at build time, like annex, so search pays only for
            # an inner product.
            self._train = self._train / np.linalg.norm(self._train, axis=1, keepdims=True)
        # ||t||^2 is constant per row; precomputing it is the standard trick
        # that turns L2 search into a single matrix product.
        self._train_sq = (self._train**2).sum(axis=1)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if k == 0:
            return np.empty((q.shape[0], 0), dtype=np.uint64)
        if self._metric == "cosine":
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dots = q @ self._train.T
        if self._metric == "l2":
            # ||q||^2 is the same for every row of a query, so it cannot change
            # the ranking and is left out.
            scores = self._train_sq[None, :] - 2.0 * dots
        else:
            scores = -dots  # 1 - dot ranks like -dot
        # argpartition finds the k best in O(n); only those k are then sorted.
        top = np.argpartition(scores, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        return np.take_along_axis(top, order, axis=1).astype(np.uint64)


# Name -> class. run.py instantiates a fresh engine per run.
ENGINES: dict[str, type[Engine]] = {
    AnnexFlat.name: AnnexFlat,
    NumpyFlat.name: NumpyFlat,
    # M1: "annex-hnsw". M2: "hnswlib", "faiss-hnsw".
}


def create(name: str) -> Engine:
    try:
        return ENGINES[name]()
    except KeyError:
        raise ValueError(f"unknown engine {name!r}; choose from {sorted(ENGINES)}") from None
