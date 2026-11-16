"""Quality and speed measurements shared by every engine."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


def recall_at_k(found: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Mean over queries of |found[:k] ∩ ground_truth[:k]| / k.

    This is the ann-benchmarks definition: the order inside the top k does not
    matter, only how many of the true k nearest were returned. Both arrays are
    (nq, >= k) label arrays.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if found.shape[0] != ground_truth.shape[0]:
        raise ValueError(f"{found.shape[0]} result rows but {ground_truth.shape[0]} ground truth rows")
    if found.shape[1] < k or ground_truth.shape[1] < k:
        raise ValueError(f"need at least k={k} columns, got {found.shape[1]} and {ground_truth.shape[1]}")
    hits = [
        np.intersect1d(f[:k], g[:k], assume_unique=True).size
        for f, g in zip(found, ground_truth)
    ]
    return float(np.mean(hits)) / k


@contextmanager
def timer():
    """Measure wall-clock time of a with-block: `with timer() as t: ...; t()`.

    perf_counter is monotonic and has the highest available resolution, unlike
    time.time(), which can jump when the system clock is adjusted.
    """
    start = time.perf_counter()
    end = None

    def elapsed() -> float:
        return (end if end is not None else time.perf_counter()) - start

    try:
        yield elapsed
    finally:
        end = time.perf_counter()


@dataclass(frozen=True)
class LatencyStats:
    qps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def latency_stats(latencies_s: np.ndarray) -> LatencyStats:
    """Summarise per-query latencies (seconds).

    QPS is queries divided by the summed query time, i.e. 1 / mean latency.
    Percentiles show the tail that a mean hides.
    """
    lat = np.asarray(latencies_s, dtype=np.float64)
    if lat.size == 0:
        raise ValueError("no latencies")
    total = lat.sum()
    p50, p95, p99 = np.percentile(lat, [50, 95, 99]) * 1e3
    return LatencyStats(
        qps=lat.size / total if total > 0 else float("inf"),
        mean_ms=float(lat.mean() * 1e3),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
    )
