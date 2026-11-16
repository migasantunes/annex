# Decision log

Short records of design choices: what was decided, why, and what was rejected.
Add an entry whenever a choice would otherwise have to be re-argued later.
Newest at the bottom.

## D1. C++20, no runtime dependencies in the core

`std::span` gives bounds-carrying views without a library. GoogleTest, Google
Benchmark and pybind11 are build-time or test-time only. A dependency-free core
is easy to embed and makes the "from scratch" claim true.

## D2. Every metric is a distance (smaller is better)

One comparison direction in all index code. Inner product is returned as
`1 - dot`, matching hnswlib, so results can be compared directly in the
benchmark harness. Rejected: returning similarities for inner product and
branching on the metric inside heaps.

## D3. Cosine is implemented as normalise-then-inner-product

Normalising once on insert and once per query removes two norm computations
from every distance evaluation. Cost: stored vectors are not the originals,
which matters only if a future API returns stored vectors.

## D4. Runtime kernel dispatch through a function pointer

The CPU is only known at runtime and wheels must run on machines other than
the build machine. Rejected: `-march=native` by default (illegal-instruction
crashes on older CPUs); templates over the kernel (cannot be chosen at
runtime).

## D5. Deterministic result order

Results are ordered by distance, then by ascending label. Without the
tie-break, tests comparing an index with the oracle are flaky whenever two
vectors are equidistant, which happens with duplicated or integer-valued data
such as SIFT.

## D6. `FlatIndex` rejects duplicate labels

A label identifies one vector. Silently keeping two would make "the neighbour
with label 7" ambiguous. Update and delete semantics are deferred to the
stretch goals.

## D7. Correctness first, then measure, then optimise

M1 uses the simplest storage that can be correct. M2 builds the measuring
equipment. Only M3 and M4 optimise, each change justified by a profile. This
ordering is the point of the project: the write-up is the story of those
measurements.
