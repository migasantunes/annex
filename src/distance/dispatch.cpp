// Runtime selection of distance kernels.
//
// The choice is made here, in one place, so index code only ever holds a
// DistanceFn and never needs to know which instruction set is in use.

#include "annex/distance.hpp"

namespace annex {
namespace {

// The kernels chosen for this process. Today there is only one set; in M3 an
// AVX2 set will join it and kernels() will pick between them.
struct KernelSet {
  const char* name;
  DistanceFn l2;
  DistanceFn inner_product;
};

constexpr KernelSet kScalarKernels{"scalar", &l2_squared_scalar, &inner_product_distance_scalar};

const KernelSet& kernels() noexcept {
  // M3: add the CPU-feature check here, e.g.
  //   if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
  //     return kAvx2Kernels;
  // Keeping it behind this one function means resolve_distance() and
  // active_kernel_name() can never disagree about what was picked.
  return kScalarKernels;
}

}  // namespace

DistanceFn resolve_distance(Metric metric) noexcept {
  const KernelSet& set = kernels();
  switch (metric) {
    case Metric::L2:
      return set.l2;
    case Metric::InnerProduct:
    case Metric::Cosine:
      // Cosine vectors are normalised before they reach the kernel, at which
      // point 1 - cos(a, b) == 1 - dot(a, b).
      return set.inner_product;
  }
  // Unreachable for valid enum values. Falling back to L2 rather than
  // returning nullptr keeps the function noexcept-safe for callers.
  return set.l2;
}

const char* active_kernel_name() noexcept { return kernels().name; }

}  // namespace annex
