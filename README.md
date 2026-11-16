# annex

A vector search engine written from scratch in C++.

Given millions of embedding vectors, find the ones nearest to a query in well
under a millisecond. This is the component underneath every retrieval-augmented
generation system and most recommenders. annex builds it from first
principles: the graph index, the SIMD distance kernels, the concurrency, the
on-disk format and the quantization, each measured against established
libraries.

> **Status: work in progress.** The foundation is in place: build system,
> exact search, Python bindings, benchmark harness and CI. The approximate
> index (HNSW) is the next milestone. No performance claims are made yet; they
> will appear here with the commands that reproduce them.

## Roadmap

| | Milestone | State |
| --- | --- | --- |
| M0 | Build system, exact `FlatIndex`, bindings, harness, CI | Done |
| M1 | HNSW graph index | Next |
| M2 | Benchmarks against hnswlib and FAISS on SIFT1M and GloVe | |
| M3 | AVX2 distance kernels with runtime dispatch | |
| M4 | Cache-aware memory layout, guided by profiling | |
| M5 | Parallel build and concurrent queries | |
| M6 | Memory-mapped persistence | |
| M7 | Scalar and product quantization | |
| M8 | Wheels, integration, write-up | |

Details and acceptance criteria: [docs/PLAN.md](docs/PLAN.md). Design:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build and test

Requires CMake 3.24 or newer, Ninja, and a C++20 compiler (GCC 12+ or Clang 15+).

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

The `dev` preset is a debug build with AddressSanitizer and
UndefinedBehaviorSanitizer. Use `release` for an optimised build with
micro-benchmarks:

```bash
cmake --preset release && cmake --build --preset release
./build/release/bench/annex_bench
```

## Python

```bash
pip install ".[test]"
pytest python/tests -q
```

```python
import numpy as np
import annex

rng = np.random.default_rng(0)
vectors = rng.standard_normal((10_000, 128), dtype=np.float32)

index = annex.FlatIndex(dim=128, metric="l2")   # "l2", "ip" or "cosine"
index.add(vectors)
labels, distances = index.search(vectors[:5], k=10)
```

## Benchmark harness

```bash
pip install ".[bench]"
python -m benchmarks.run --engine annex-flat --dataset synthetic \
    --n 20000 --dim 64 --queries 200 --k 10
```

## What annex is not

It is a library, not a server: no network protocol, no query language, no
replication. See the architecture document for the reasoning.

## License

MIT
