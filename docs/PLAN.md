# Work plan

Nine milestones. Each one ends with something that builds, is tested, and can
be shown. Do them in order: every milestone depends on the one before it,
except where noted.

Work on a branch per milestone (`m1-hnsw`, `m2-benchmarks`, ...) and merge
through a pull request so CI runs on it. The pull request description is where
the measurements for that milestone go.

**Definition of done, for every milestone**

1. The acceptance criteria below are met and the evidence is in the PR.
2. `dev`, `release` and (from M5) `tsan` presets pass, and so do the Python
   tests.
3. You can explain every function you merged without looking at it. If you
   cannot, the milestone is not done.
4. `docs/ARCHITECTURE.md` matches the code, and any design change is logged in
   `docs/DECISIONS.md`.

Targets marked *(target)* are estimates of what a good implementation reaches.
They are not promises. If a target is missed, find out why and write that
down: an explained gap is worth more in an interview than a lucky number.

---

## M0. Walking skeleton (done)

Everything needed so that later milestones are only about the engine.

- [x] CMake project, presets (`dev`, `release`, `tsan`), strict warnings
- [x] Scalar distance kernels and runtime dispatch point
- [x] `Index` interface and exact `FlatIndex`
- [x] GoogleTest suite including a randomised check against a naive reference
- [x] Google Benchmark micro-benchmarks
- [x] Python bindings and pytest suite
- [x] Benchmark harness that runs end to end on synthetic data
- [x] CI: format check, gcc and clang, sanitizers, Python

---

## M1. HNSW, single-threaded

The core of the project. Design: ARCHITECTURE.md §5.

**Before coding:** read sections 3 and 4 of the HNSW paper and write, in your
own words, a half-page note in `docs/notes/hnsw.md` answering: why layers, what
`ef` controls, and why the neighbour-selection heuristic beats "keep the M
closest".

**Tasks**

1. `include/annex/hnsw_index.hpp` and `src/hnsw_index.cpp` with the simple
   storage design.
2. Level draw with a seeded RNG, so builds are reproducible.
3. `search_layer` (Alg. 2). Test it alone on a tiny hand-built graph.
4. `select_neighbors_heuristic` (Alg. 4). Test on a hand-checkable 2D case.
5. `insert` (Alg. 1), including pruning a neighbour's list when it overflows.
6. `search` (Alg. 5) and `set_ef_search`.
7. Python bindings for `HnswIndex`, and an `annex-hnsw` engine in the harness.

**Tests to write**

- Contract tests shared with `FlatIndex` (same throws, same ordering). Turn the
  existing contract tests into a typed test suite that runs on both.
- Graph invariants after a build: no node exceeds its link limit, no self
  links, no duplicate links, every link points to an existing node, every node
  is reachable from the entry point on layer 0.
- Recall floor: on 10k random 32-d vectors, recall@10 against `FlatIndex` is at
  least 0.95 with `ef_search = 100`.
- Determinism: same seed and insertion order gives identical results.

**Acceptance**

- All of the above pass under ASan and UBSan.
- With `ef_search` equal to the number of vectors, results equal `FlatIndex`
  on a connected graph (a useful sanity check that search is not dropping
  candidates).

---

## M2. Benchmarks against hnswlib and FAISS

Measure before optimising. This milestone produces the baseline curve that
every later milestone is judged against.

**Tasks**

1. Dataset loader for ann-benchmarks HDF5 files (`sift-128-euclidean`,
   `glove-100-angular`), downloaded into `benchmarks/data/` (git-ignored).
   Add `h5py` to the `bench` extra.
2. Engine adapters for `hnswlib` and `faiss-cpu` (`IndexHNSWFlat`), behind the
   same interface as the annex engines.
3. Sweep `ef_search` over about ten values and write one JSON result file per
   run, including the environment block (ARCHITECTURE.md §9).
4. `benchmarks/plot.py`: recall@10 against QPS, log-scale y axis, one line per
   engine.

**Acceptance**

- One command reproduces the SIFT1M curve for all three engines.
- At the same `M`, `ef_construction` and `ef_search`, annex's recall@10 is
  within 0.01 of hnswlib's. Speed is expected to be far behind at this point,
  and the PR should state by how much.
- The plot and the result JSON are committed under `benchmarks/results/baseline/`.

---

## M3. SIMD kernels

Design: ARCHITECTURE.md §6.1.

**Tasks**

1. `src/distance/avx2.cpp` with L2 and inner-product kernels, compiled with
   per-file flags.
2. CPU detection in `dispatch.cpp`; `active_kernel_name()` returns `"avx2"`.
3. An environment variable (`ANNEX_FORCE_SCALAR=1`) to force the scalar path,
   so both can be benchmarked from one binary.
4. Tests comparing AVX2 with scalar for every dim from 0 to 70 and a few large
   ones, with a relative tolerance.

**Acceptance**

- Kernel micro-benchmark shows the AVX2 speedup at dims 96 to 1536; report the
  table. *(target: 3x or more at dim 128 and above. Note that the compiler may
  auto-vectorise the scalar loop with SSE, so check the assembly to know what
  you are comparing against.)*
- End-to-end QPS on SIFT1M improves over the M2 baseline; report the new curve.
- Tests pass with and without `ANNEX_FORCE_SCALAR`.

---

## M4. Memory layout and profiling

Design: ARCHITECTURE.md §6.2. This is the milestone that teaches the most
about real performance work.

**Tasks**

1. Profile the M3 build with `perf record` / `perf report` on the SIFT1M query
   workload. Save the top of the report in the PR. Find out what fraction of
   time is distance computation, heap operations, and visited-set work, and how
   many cache misses there are per query (`perf stat -e cache-misses`).
2. Flat contiguous layer-0 storage.
3. Pooled epoch-tagged visited list.
4. Software prefetch.
5. Re-profile after each change. Keep changes that help and revert those that
   do not, recording both in the PR.

**Acceptance**

- A before/after table of QPS at 0.95 recall for each change.
- *(target: within a factor of 1.5 of hnswlib single-threaded QPS at 0.95
  recall@10 on SIFT1M. hnswlib has had years of tuning; closing most of the gap
  is the result.)*

---

## M5. Concurrency

Design: ARCHITECTURE.md §6.3.

**Tasks**

1. Thread-safe `add` with per-node locks and a global entry-point lock.
2. `add_batch(labels, vectors, num_threads)` in C++ and the bindings.
3. Parallel `search` over a batch of queries in the bindings.
4. Stress tests: concurrent inserts, concurrent queries, and both together if
   that is supported.

**Acceptance**

- All tests pass under the `tsan` preset with zero reports.
- Graph invariants from M1 hold after a parallel build.
- Recall after a parallel build is within 0.01 of a single-threaded build.
- Build-time scaling table for 1, 2, 4 and 8 threads.

---

## M6. Persistence

Design: ARCHITECTURE.md §6.4.

**Tasks**

1. Write the file format specification in `docs/FORMAT.md` first.
2. `save(path)` and a static `load(path)` using `mmap`.
3. Header validation: magic, version, sizes consistent with file length,
   checksum.
4. Python `save` / `load`.

**Acceptance**

- Round trip: a loaded index returns results identical to the original.
- Truncated, zero-length, wrong-magic and bit-flipped files all raise a clear
  error and never crash, including under ASan.
- Load time for the SIFT1M index is reported.

---

## M7. Quantization

Design: ARCHITECTURE.md §6.5. Independent of M5 and M6; can be done earlier.

**Tasks**

1. SQ8 storage with a `uint8` kernel (scalar, then AVX2).
2. k-means and a PQ codec with asymmetric distance computation.
3. Rerank step using full-precision vectors.
4. Harness engines for each variant.

**Acceptance**

- A table of bytes per vector, recall@10 and QPS for float32, SQ8, and PQ with
  and without reranking, on SIFT1M.

---

## M8. Packaging and write-up

**Tasks**

1. Wheels built in CI with `cibuildwheel` for Linux x86-64. Check that the
   name `annex` is free on PyPI before publishing; rename the distribution if
   not.
2. Use annex as the retriever in `eurlex-rag` behind a flag, and compare
   retrieval latency with what it uses now.
3. README: final curves, the numbers table, a short "how it works", and the
   list of what annex does not do.
4. A technical write-up of the optimisation journey from M2 to M4, with the
   profiles. This is the LinkedIn post and the interview story.

**Acceptance**

- `pip install` from a built wheel works in a clean environment.
- Every number in the README can be regenerated with a documented command.

---

## Stretch goals

Only after M8, and only if they are interesting to you:

- Deletion with tombstones and link repair.
- Filtered search (a predicate on labels).
- AVX-512 and ARM NEON kernels.
- An IVF index as a second approximate method for comparison.
