# Benchmark harness

Measures an engine end to end: build an index, run queries, compare the answers
with exact ground truth. Methodology: [ARCHITECTURE.md §9](../docs/ARCHITECTURE.md).

## Running

From the repository root (the harness is not part of the installed package):

```bash
pip install ".[bench]"
python -m benchmarks.run --engine annex-flat --dataset synthetic --n 2000 --dim 32 --queries 100 --k 10
python -m benchmarks.run --help
```

Each run prints a summary and writes a JSON file to
`benchmarks/results/<dataset>-<engine>-<timestamp>.json` (git-ignored), or to
`--output PATH`.

## What the numbers mean

| Field | Meaning |
| --- | --- |
| `recall` | recall@k: for each query, the fraction of the true k nearest neighbours that the engine returned, averaged over queries. Order inside the top k is ignored. Exact engines must score 1.0. |
| `qps` | Queries per second, **single thread, one query at a time** (the ann-benchmarks convention): number of queries divided by the summed time of the individual `search` calls. |
| `latency_ms` | Mean, p50, p95 and p99 of the per-query times. The tail shows stalls that the mean hides. |
| `build_seconds` | Wall-clock time to build the index from the training vectors. |
| `environment` | CPU, Python, NumPy and annex versions, annex's active distance kernel, git commit (`-dirty` if there were uncommitted changes), BLAS thread settings, UTC time. Without it a number cannot be reproduced. |

Engines today:

- `annex-flat`: `annex.FlatIndex`, exact search in C++.
- `numpy-flat`: exact search in NumPy (one matrix product per call). The baseline
  the C++ code has to beat. BLAS is pinned to one thread so both run on one core.

Datasets today: `synthetic`, seeded Gaussian vectors with exact ground truth
computed by NumPy brute force. Random Gaussian data is much easier for ANN
indexes than real embeddings, so use it for smoke tests, not for headline
numbers.

## Adding an engine

1. In `engines.py`, subclass `Engine` and implement `build(train, metric)` and
   `search(queries, k)`, returning `(nq, k)` uint64 labels where the label is
   the row number in `train`. Approximate engines also override
   `set_query_param` (e.g. `ef_search`).
2. Import optional libraries inside `build`, so the harness still runs without
   them.
3. Register the class in `ENGINES` and add its library to the `bench` extra in
   `pyproject.toml`.

## Adding a dataset

1. In `datasets.py`, write a loader that accepts the keyword arguments
   `n, dim, queries, k, metric, seed` (ignoring those that do not apply) and
   returns a `Dataset`.
2. Register it in `DATASETS`. Downloaded files go in `benchmarks/data/`
   (git-ignored).
