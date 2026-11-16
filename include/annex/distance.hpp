// Distance kernels and their runtime dispatch.
//
// The kernels are the hottest code in the engine: an HNSW query evaluates
// hundreds to thousands of them. Index code never calls a kernel by name; it
// asks resolve_distance() once at construction and keeps the function pointer.
// That single indirection is what lets milestone M3 add AVX2 kernels without
// touching any index code.
#pragma once

#include <cstddef>

#include "annex/types.hpp"

namespace annex {

/// Signature shared by every kernel. `a` and `b` each point at `dim` floats.
/// No alignment is assumed.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

// --- Portable reference kernels -------------------------------------------
// Plain loops, always available, and the oracle the SIMD kernels are tested
// against.

/// Squared Euclidean distance: sum((a[i] - b[i])^2).
[[nodiscard]] float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept;

/// Inner-product distance: 1 - dot(a, b).
[[nodiscard]] float inner_product_distance_scalar(const float* a, const float* b,
                                                  std::size_t dim) noexcept;

// --- Dispatch ---------------------------------------------------------------

/// Best available kernel for `metric` on the CPU we are running on.
/// Cosine resolves to the inner-product kernel: callers normalise vectors
/// first (see normalize()).
[[nodiscard]] DistanceFn resolve_distance(Metric metric) noexcept;

/// Name of the instruction set resolve_distance() picked, e.g. "scalar".
/// Recorded in benchmark output so results are reproducible.
[[nodiscard]] const char* active_kernel_name() noexcept;

/// Scale `v` in place to unit L2 norm. A zero vector is left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

}  // namespace annex
