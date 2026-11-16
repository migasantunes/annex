# annex

A vector search engine (approximate nearest neighbour) written from scratch in
C++20, with Python bindings and a benchmark harness. It is a portfolio project:
the owner, Miguel, must be able to explain and defend every line in an
interview.

## Read first

- `docs/PLAN.md`: the milestones, their tasks and acceptance criteria. Work on
  one milestone at a time, in order.
- `docs/ARCHITECTURE.md`: the design, including the planned design for each
  milestone.
- `docs/DECISIONS.md`: why things are the way they are. Add to it when a design
  choice is made or changed.

## How to work in this repo

1. **Explain before implementing.** At the start of a task, state the idea and
   the plan in plain language and wait for Miguel to confirm he follows it. For
   anything from a paper, point at the exact section or algorithm number.
2. **Small steps.** One function or one concept per change, with its test.
   Never deliver a whole milestone in one go.
3. **Test against the oracle.** `FlatIndex` is always right. Any approximate
   index is checked against it.
4. **Measure before and after any optimisation.** No performance change is
   merged without numbers from `bench/` or `benchmarks/`. If a change does not
   help, revert it and note that it was tried.
5. **No invented numbers.** Every figure in the README, a PR or a doc comes
   from a command that is written next to it.
6. **Do not weaken checks to get green.** Fix the cause. Sanitizer and TSan
   reports are bugs.
7. One branch and one pull request per milestone; CI must pass before merging.

## Commands

```bash
# C++: debug build with ASan + UBSan, warnings as errors
cmake --preset dev && cmake --build --preset dev && ctest --preset dev

# C++: optimised build with micro-benchmarks
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/bench/annex_bench

# C++: ThreadSanitizer (matters from M5)
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan

# Run one test
./build/dev/tests/annex_tests --gtest_filter='FlatIndex*'

# Python: build the extension, run tests
pip install ".[test]"
pytest python/tests -q

# End-to-end benchmark harness
pip install ".[bench]"
python -m benchmarks.run --engine annex-flat --dataset synthetic --n 20000 --dim 64 --queries 200 --k 10

# Format
clang-format -i $(git ls-files '*.hpp' '*.cpp')
```

## Layout

| Path | Contents |
| --- | --- |
| `include/annex/` | Public headers (the API) |
| `src/`, `src/distance/` | Implementations and distance kernels |
| `tests/` | GoogleTest suite (`annex_tests`) |
| `bench/` | Google Benchmark micro-benchmarks (`annex_bench`) |
| `python/` | pybind11 bindings, `annex` package, pytest tests |
| `benchmarks/` | Python harness: datasets, engines, recall/QPS, plots |
| `docs/` | Plan, architecture, decisions |

## Conventions

- C++20, Google style, 100 columns, `#pragma once`, everything in namespace
  `annex`. Members end in `_`. Formatting is enforced by `.clang-format`.
- Every metric is a distance: smaller is better. Results are ordered with
  `Neighbor::operator<` (distance, then label).
- Public functions validate arguments and throw `std::invalid_argument`. Hot
  internal paths do not validate; they `assert`.
- No `-march=native` by default. SIMD code lives in its own translation unit
  with per-file flags and is selected at runtime (ARCHITECTURE.md §4.2).
- No new runtime dependency in the C++ core.
- Comments explain why, not what.
- Python bindings: NumPy in and out, GIL released around core calls,
  `ValueError` for bad arguments.
