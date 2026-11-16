# Architecture

This document describes how annex is put together and why. Sections marked
**(built)** describe code that exists today; sections marked **(planned, Mn)**
are the design for a later milestone in [PLAN.md](PLAN.md) and are a starting
point, not a constraint. If implementation shows a planned design is wrong,
change the design and record why in [DECISIONS.md](DECISIONS.md).

## 1. What annex is

An in-process library that stores fixed-dimension `float32` vectors and answers
"which k stored vectors are nearest to this query?". The core is C++20 with no
runtime dependencies. A thin pybind11 layer exposes it to Python, and a Python
harness measures it against established libraries.

It is deliberately **not** a database server: no network protocol, no query
language, no replication. Those are different engineering problems. The
interesting part of a vector database is the index, so that is the whole scope.

## 2. Layers

```
┌─────────────────────────────────────────────────────────────┐
│ benchmarks/        Python harness: datasets, engines,       │
│                    recall + QPS measurement, plots          │
├─────────────────────────────────────────────────────────────┤
│ python/            pybind11 bindings (annex._core) and the  │
│                    `annex` package; NumPy in, NumPy out     │
├─────────────────────────────────────────────────────────────┤
│ include/annex/     Index interface                          │
│ src/               FlatIndex (built) · HnswIndex (M1)       │
├─────────────────────────────────────────────────────────────┤
│ src/distance/      kernels: scalar (built) · AVX2 (M3)      │
│                    chosen once at runtime by dispatch       │
└─────────────────────────────────────────────────────────────┘
```

Dependencies only point downwards. The C++ core knows nothing about Python.

## 3. Repository layout

| Path | Contents |
| --- | --- |
| `include/annex/` | Public headers. The API surface. |
| `src/` | Implementations. `src/distance/` holds the kernels. |
| `tests/` | GoogleTest unit and property tests (`annex_tests`). |
| `bench/` | Google Benchmark micro-benchmarks (`annex_bench`): kernels and single-index hot paths. |
| `python/` | Bindings (`python/src/`), the `annex` package, and pytest tests. |
| `benchmarks/` | End-to-end Python harness comparing engines on datasets. |
| `docs/` | This file, the plan, and the decision log. |

`bench/` and `benchmarks/` answer different questions. `bench/` asks "how many
nanoseconds does this function take?" and guides optimisation. `benchmarks/`
asks "at 95% recall, how many queries per second, compared with hnswlib?" and
produces the numbers that go in the README.

## 4. Core concepts (built)

### 4.1 Distances

Every metric is a *distance*: smaller is more similar. This lets all index code
use one comparison direction.

| Metric | Value returned | Note |
| --- | --- | --- |
| `L2` | `sum((a-b)^2)` | Squared: the square root does not change the ranking, so we never pay for it. |
| `InnerProduct` | `1 - dot(a, b)` | Same convention as hnswlib, which makes results directly comparable. |
| `Cosine` | `1 - cos(a, b)` | Vectors are normalised when added and when queried, then the inner-product kernel is used. |

### 4.2 Kernel dispatch

A kernel has the signature `float(const float*, const float*, size_t) noexcept`.
An index calls `resolve_distance(metric)` once in its constructor and stores the
returned function pointer.

Why a function pointer and not a template parameter or a virtual call:

- The choice depends on the **CPU the binary runs on**, which is only known at
  runtime. A wheel built on one machine must work on another.
- The base library is compiled for baseline x86-64. In M3 the AVX2 kernels go in
  their own translation unit compiled with `-mavx2 -mfma`, and dispatch picks
  them only after checking CPU support. Compiling everything with
  `-march=native` would be faster to write and would crash with an illegal
  instruction on older CPUs.
- An indirect call costs about a nanosecond against a kernel that takes tens of
  nanoseconds. Measure this in M4 before considering anything fancier.

### 4.3 The `Index` interface

`add(label, vector)` and `search(query, k)`, plus `size()`, `dim()`,
`metric()`. The exact contract (what throws, how ties are ordered) is written in
`include/annex/index.hpp` and enforced by tests. Results are ordered by
`Neighbor::operator<`: nearest first, ties broken by ascending label, so every
index is deterministic and directly comparable with the oracle.

### 4.4 `FlatIndex`

Brute force over one contiguous `std::vector<float>`. It keeps the best k seen
so far in a bounded max-heap: the heap's top is the *worst* of the current best
k, so a new candidate only needs one comparison to be rejected.

It is the correctness oracle. Every approximate index is tested by comparing
its results with `FlatIndex` on the same data.

## 5. HNSW (planned, M1)

Reference: Malkov & Yashunin, *Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs* (arXiv
1603.09320). Read sections 3 and 4 before writing code; Algorithms 1 to 5 map
almost one-to-one onto functions.

**Idea.** Build a graph where each vector is a node linked to some of its near
neighbours. To search, start somewhere and greedily walk towards the query.
A single-layer graph gets stuck taking many small steps, so HNSW stacks layers:
upper layers contain exponentially fewer nodes with longer links (an express
network), and layer 0 contains everything.

**Proposed public API**

```cpp
struct HnswParams {
  std::size_t m = 16;                 // max links per node on layers > 0
  std::size_t ef_construction = 200;  // candidate list size while building
  std::uint64_t seed = 42;            // level assignment RNG
};

class HnswIndex final : public Index {
 public:
  HnswIndex(std::size_t dim, Metric metric, HnswParams params = {});
  void add(Label label, std::span<const float> vector) override;
  std::vector<Neighbor> search(std::span<const float> query, std::size_t k) const override;
  void set_ef_search(std::size_t ef);  // recall/speed knob at query time
  // ...
};
```

**Internal pieces**

| Piece | Paper | Responsibility |
| --- | --- | --- |
| level draw | §4, `l = floor(-ln(U) * mL)` with `mL = 1/ln(M)` | Picks each node's top layer. |
| `search_layer` | Alg. 2 | Best-first search on one layer with a candidate min-heap and a result max-heap of size `ef`. |
| `select_neighbors_heuristic` | Alg. 4 | Chooses which candidates become links. Keeps a candidate only if it is closer to the new node than to any already-selected neighbour, which preserves links across clusters. Use this, not Alg. 3: it matters for recall on clustered data. |
| `insert` | Alg. 1 | Greedy descent to the node's top layer, then `search_layer` + select + bidirectional linking + pruning on each layer down to 0. |
| `search` | Alg. 5 | Greedy descent with `ef = 1`, then `search_layer` on layer 0 with `ef = max(ef_search, k)`. |

Layer 0 allows `2 * M` links, upper layers `M`.

**First storage design (M1): simple and correct.** Per node, a
`std::vector<std::vector<std::uint32_t>>` of neighbour lists, and a
`std::vector<bool>` visited set per query. This is slow on purpose. Get recall
right first; M4 replaces the storage once profiling shows where time goes.

Internal node ids are dense `uint32_t` slots. Labels map to slots through one
table; nothing inside the graph uses labels.

## 6. Planned designs for later milestones

### 6.1 AVX2 kernels (M3)

`src/distance/avx2.cpp`, compiled with `-mavx2 -mfma` through
`set_source_files_properties`. Eight floats per iteration with
`_mm256_fmadd_ps`, a horizontal sum at the end, and a scalar loop for the
remaining `dim % 8` elements. Dispatch uses `__builtin_cpu_supports("avx2")`.
Tests compare every SIMD kernel with the scalar one across all dims from 0 to
about 70, to catch tail-handling bugs, using a relative tolerance because
floating-point summation order differs.

### 6.2 Memory layout (M4)

Graph traversal is dominated by cache misses, not arithmetic. Planned changes,
each justified by a `perf` profile before and after:

- One contiguous block per node on layer 0: `[link count | links... | vector]`,
  so reading a node's links and its vector touches adjacent memory.
- `_mm_prefetch` the next candidate's block while computing the current
  distance.
- Replace `std::vector<bool>` with a pooled visited list of `uint16_t` epoch
  tags, so resetting between queries is an increment, not a clear.

### 6.3 Concurrency (M5)

- Parallel build: one mutex per node guarding its link lists, plus a global
  lock for entry-point and max-level changes (the hnswlib approach).
- Queries take no global lock. Decide and document whether queries may run
  during inserts; if yes, readers lock the node whose links they copy.
- Everything is validated under the `tsan` preset. A data race is a bug even
  when the test passes.

### 6.4 Persistence (M6)

A single file: fixed header (magic bytes, format version, dim, metric, params,
counts, checksum), then the layer-0 block, then upper-layer links, then the
label table. Loading is `mmap`, so a large index opens instantly and pages in
on demand. The header is validated before anything else is trusted; a corrupt
or truncated file must produce an error, never a crash.

### 6.5 Quantization (M7)

- **SQ8:** each dimension mapped to one byte using per-dimension min/max. 4x
  less memory, with a dedicated `uint8` kernel.
- **PQ:** split the vector into `m` sub-vectors, run k-means (256 centroids) on
  each, store `m` bytes per vector. Query with asymmetric distance computation:
  precompute an `m x 256` table per query, then a distance is `m` table lookups.
  Reference: Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor
  Search* (2011).
- Search on compressed codes, then **rerank** the top candidates with the full
  vectors. Report memory, recall and QPS for each variant.

## 7. Python API

Built today for `FlatIndex`; `HnswIndex` follows the same shape in M1.

```python
import numpy as np
import annex

index = annex.FlatIndex(dim=128, metric="l2")   # "l2" | "ip" | "cosine"
index.add(vectors)                               # (n, dim) float32; labels continue from len(index)
index.add(more, labels=np.array([...], dtype=np.uint64))
labels, distances = index.search(queries, k=10)  # (nq, k) uint64 and float32
len(index); index.dim; index.metric
annex.__version__; annex.active_kernel_name()
```

Rules for the bindings:

- Accept C-contiguous `float32` arrays without copying; convert anything else.
  A single 1-D vector of length `dim` is treated as one row, and `search`
  still returns `(1, k)` arrays for it.
- Release the GIL around every loop that calls into the core. Because the GIL
  no longer serialises callers, the binding class owns a reader-writer lock:
  `search` shares it, `add` takes it exclusively.
- Bad arguments raise `ValueError` (`std::invalid_argument` from the core is
  translated automatically).
- `search` requires `k <= len(index)` so the output is rectangular.
- Labels must be non-negative integers and distinct. A batch is validated
  before anything is added, with one exception: a label that is already in the
  index is only detected when its row is reached, and earlier rows of that
  batch stay added.

## 8. Testing strategy

| Level | Tool | What it proves |
| --- | --- | --- |
| Unit | GoogleTest | Kernels and index contracts on hand-checked cases. |
| Property | GoogleTest + fixed seeds | `FlatIndex` equals a naive reference; later, HNSW recall against `FlatIndex` stays above a floor. |
| Memory safety | `dev` preset (ASan + UBSan) | No out-of-bounds, use-after-free or undefined behaviour in tests. |
| Races | `tsan` preset | No data races (meaningful from M5). |
| Bindings | pytest | Shapes, dtypes, errors, agreement with NumPy brute force. |
| End to end | `benchmarks/` | Recall and throughput on real datasets. |

## 9. Benchmark methodology

Numbers are only useful if they are comparable and reproducible.

- **Datasets:** the standard ann-benchmarks HDF5 files, which ship with query
  sets and exact ground truth: `sift-128-euclidean` (1M vectors) and
  `glove-100-angular` (about 1.2M).
- **Metric:** recall@10 against the dataset's ground truth, and queries per
  second, single-threaded, one query at a time. Multi-threaded numbers are
  reported separately and labelled.
- **Curve, not a point:** sweep `ef_search` and plot recall against QPS. A
  single number hides the trade-off.
- **Same parameters for every engine:** `M` and `ef_construction` identical for
  annex and hnswlib.
- **Record the environment** in every result file: CPU model, compiler and
  flags, active kernel (`active_kernel_name()`), library versions, git commit.
- **Never publish a number the harness did not produce.**
