// End-to-end brute-force query latency: the baseline every ANN index must beat.

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "annex/distance.hpp"
#include "annex/flat_index.hpp"
#include "annex/types.hpp"

namespace {

constexpr std::size_t kDim = 128;
constexpr std::size_t kK = 10;
constexpr std::size_t kNumQueries = 64;

void BM_FlatSearch(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  // Fixed seed so every run benchmarks the same data.
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  annex::FlatIndex index(kDim, annex::Metric::L2);
  index.reserve(n);
  std::vector<float> v(kDim);
  for (std::size_t i = 0; i < n; ++i) {
    for (float& x : v) {
      x = dist(rng);
    }
    index.add(static_cast<annex::Label>(i), v);
  }

  // Rotate through several queries so the result isn't one cached answer.
  std::vector<std::vector<float>> queries(kNumQueries, std::vector<float>(kDim));
  for (auto& q : queries) {
    for (float& x : q) {
      x = dist(rng);
    }
  }

  std::size_t qi = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(index.search(queries[qi], kK));
    qi = (qi + 1) % kNumQueries;
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));  // queries/s
  state.SetLabel(annex::active_kernel_name());
}

BENCHMARK(BM_FlatSearch)
    ->ArgName("n")
    ->Arg(1'000)
    ->Arg(10'000)
    ->Arg(100'000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
