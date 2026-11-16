// Portable reference kernels.
//
// These are deliberately plain loops: they are the oracle the SIMD kernels are
// tested against, so their correctness must be obvious by inspection. The
// compiler may still auto-vectorise them for the baseline ISA (SSE2).

#include <cmath>
#include <cstddef>

#include "annex/distance.hpp"

namespace annex {

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float inner_product_distance_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float dot = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return 1.0F - dot;
}

void normalize(float* v, std::size_t dim) noexcept {
  // Accumulate in double: this runs once per vector (not per comparison), so
  // the extra precision is free and keeps unit vectors as close to norm 1 as
  // float allows, which is what makes cosine == inner product hold afterwards.
  double norm_sq = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    norm_sq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
  }
  // A zero vector has no direction; dividing would produce NaNs that then
  // poison every distance computed against it. Leave it as is.
  if (norm_sq == 0.0) {
    return;
  }
  const double inv_norm = 1.0 / std::sqrt(norm_sq);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = static_cast<float>(static_cast<double>(v[i]) * inv_norm);
  }
}

}  // namespace annex
