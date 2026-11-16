// Per-call cost of each distance kernel across common embedding sizes
// (96: Deep1B, 128: SIFT, 384/768: sentence transformers, 1536: OpenAI).

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "annex/distance.hpp"
#include "annex/types.hpp"

namespace {

std::vector<float> random_vector(std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(dim);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

void run_kernel(benchmark::State& state, annex::DistanceFn fn) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const std::vector<float> a = random_vector(dim, 1);
  const std::vector<float> b = random_vector(dim, 2);
  for (auto _ : state) {
    // DoNotOptimize on the result stops the call being hoisted or deleted;
    // the inputs are opaque because they come through a function pointer.
    benchmark::DoNotOptimize(fn(a.data(), b.data(), dim));
  }
  // Two input floats read per dimension per call.
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(2 * dim * sizeof(float)));
}

void BM_L2SquaredScalar(benchmark::State& state) { run_kernel(state, &annex::l2_squared_scalar); }

void BM_InnerProductScalar(benchmark::State& state) {
  run_kernel(state, &annex::inner_product_distance_scalar);
}

void BM_L2Dispatched(benchmark::State& state) {
  run_kernel(state, annex::resolve_distance(annex::Metric::L2));
  state.SetLabel(annex::active_kernel_name());
}

void BM_InnerProductDispatched(benchmark::State& state) {
  run_kernel(state, annex::resolve_distance(annex::Metric::InnerProduct));
  state.SetLabel(annex::active_kernel_name());
}

void dims(benchmark::internal::Benchmark* b) {
  for (const std::int64_t dim : {96, 128, 384, 768, 1536}) {
    b->Arg(dim);
  }
  b->ArgName("dim");
}

BENCHMARK(BM_L2SquaredScalar)->Apply(dims);
BENCHMARK(BM_InnerProductScalar)->Apply(dims);
BENCHMARK(BM_L2Dispatched)->Apply(dims);
BENCHMARK(BM_InnerProductDispatched)->Apply(dims);

}  // namespace
