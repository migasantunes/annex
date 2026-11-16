"""Run one engine on one dataset and record recall, speed and environment.

Example (also the CI smoke test):

    python -m benchmarks.run --engine annex-flat --dataset synthetic \
        --n 2000 --dim 32 --queries 100 --k 10
"""

from __future__ import annotations

import os

# BLAS libraries start a thread pool when NumPy is imported, so this has to run
# first. Without it numpy-flat would use every core while annex uses one, and
# the comparison would be about thread counts rather than the indexes. An
# explicit setting in the environment wins and is recorded in the result.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks import datasets, engines, metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--engine", required=True, choices=sorted(engines.ENGINES))
    p.add_argument("--dataset", required=True, choices=sorted(datasets.DATASETS))
    p.add_argument("--n", type=int, default=10_000, help="vectors to index (synthetic)")
    p.add_argument("--dim", type=int, default=64, help="vector dimension (synthetic)")
    p.add_argument("--queries", type=int, default=100, help="number of queries (synthetic)")
    p.add_argument("--k", type=int, default=10, help="neighbours per query; recall@k")
    p.add_argument("--metric", default="l2", choices=["l2", "ip", "cosine"])
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    p.add_argument("--output", type=Path, help="result JSON path (default: benchmarks/results/)")
    args = p.parse_args(argv)
    for name in ("n", "dim", "queries", "k"):
        if getattr(args, name) <= 0:
            p.error(f"--{name} must be positive")
    if args.k > args.n:
        p.error("--k must be <= --n")
    return args


def cpu_model() -> str | None:
    # platform.processor() is often empty on Linux; /proc/cpuinfo is reliable.
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # A result from uncommitted code cannot be reproduced from the hash alone.
    return out + ("-dirty" if dirty else "")


def environment() -> dict:
    """What a reader needs to judge or reproduce a number (ARCHITECTURE.md §9)."""
    try:
        import annex

        annex_version, kernel = annex.__version__, annex.active_kernel_name()
    except ImportError:
        annex_version = kernel = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "annex": annex_version,
        "annex_kernel": kernel,
        "git_commit": git_commit(),
        "blas_threads": {
            v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run(args: argparse.Namespace) -> dict:
    data = datasets.load(
        args.dataset,
        n=args.n,
        dim=args.dim,
        queries=args.queries,
        k=args.k,
        metric=args.metric,
        seed=args.seed,
    )
    engine = engines.create(args.engine)

    with metrics.timer() as build_time:
        engine.build(data.train, data.metric)

    # One query at a time on a single thread: the ann-benchmarks convention.
    # It measures per-query latency, which is what an online service sees, and
    # stops engines that batch or parallelise well from looking better for a
    # reason unrelated to the index itself. Batched and multi-threaded numbers
    # are a separate measurement (M5) and must be labelled as such.
    nq = data.queries.shape[0]
    found = np.empty((nq, args.k), dtype=np.uint64)
    latencies = np.empty(nq, dtype=np.float64)
    for i in range(nq):
        query = data.queries[i]
        start = time.perf_counter()
        labels = engine.search(query, args.k)
        latencies[i] = time.perf_counter() - start
        found[i] = labels[0]

    stats = metrics.latency_stats(latencies)
    return {
        "engine": args.engine,
        "dataset": data.name,
        "parameters": {
            "n": data.train.shape[0],
            "dim": data.dim,
            "queries": nq,
            "k": args.k,
            "metric": data.metric,
            "seed": args.seed,
        },
        "build_seconds": build_time(),
        "qps": stats.qps,
        "latency_ms": {
            "mean": stats.mean_ms,
            "p50": stats.p50_ms,
            "p95": stats.p95_ms,
            "p99": stats.p99_ms,
        },
        "recall": metrics.recall_at_k(found, data.ground_truth, args.k),
        "environment": environment(),
    }


def default_output(result: dict) -> Path:
    # Microseconds, so back-to-back runs (e.g. a metric sweep) never overwrite
    # each other's files.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return RESULTS_DIR / f"{result['dataset']}-{result['engine']}-{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)

    output = args.output or default_output(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    p = result["parameters"]
    lat = result["latency_ms"]
    print(f"engine   {result['engine']}  (annex kernel: {result['environment']['annex_kernel']})")
    print(f"dataset  {result['dataset']}  n={p['n']} dim={p['dim']} queries={p['queries']} metric={p['metric']}")
    print(f"build    {result['build_seconds']:.3f} s")
    print(f"recall@{p['k']} {result['recall']:.4f}")
    print(f"qps      {result['qps']:.1f}  (single thread, one query at a time)")
    print(f"latency  mean {lat['mean']:.3f} ms  p50 {lat['p50']:.3f}  p95 {lat['p95']:.3f}  p99 {lat['p99']:.3f}")
    print(f"wrote    {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
