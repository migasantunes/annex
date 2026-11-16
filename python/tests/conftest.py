"""Shared fixtures and the NumPy reference that the bindings are checked against."""

import numpy as np
import pytest


def reference_search(train, queries, k, metric):
    """Exact k-NN in NumPy, using the same distance conventions as the core.

    Ties are broken by ascending label, like Neighbor::operator< in C++, so the
    labels can be compared exactly. Labels here are row positions.
    """
    train = train.astype(np.float64)
    queries = queries.astype(np.float64)
    if metric == "l2":
        dist = ((queries[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
    elif metric == "ip":
        dist = 1.0 - queries @ train.T
    elif metric == "cosine":
        t = train / np.linalg.norm(train, axis=1, keepdims=True)
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        dist = 1.0 - q @ t.T
    else:
        raise ValueError(metric)
    # lexsort sorts by the last key first: distance, then label.
    labels = np.broadcast_to(np.arange(train.shape[0]), dist.shape)
    order = np.lexsort((labels, dist), axis=1)[:, :k]
    return order.astype(np.uint64), np.take_along_axis(dist, order, axis=1)


@pytest.fixture
def rng():
    # A fixed seed per test, so a failure always reproduces.
    return np.random.default_rng(12345)
