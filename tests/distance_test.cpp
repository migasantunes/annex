#include "annex/distance.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

#include "annex/types.hpp"

namespace annex {
namespace {

// Small inputs are exactly representable and their sums are exact in float,
// so these comparisons can be exact rather than approximate.

TEST(L2SquaredScalar, HandComputed) {
  const std::array<float, 3> a{1.0F, 2.0F, 3.0F};
  const std::array<float, 3> b{4.0F, 6.0F, 3.0F};
  // (1-4)^2 + (2-6)^2 + 0 = 9 + 16 = 25
  EXPECT_EQ(l2_squared_scalar(a.data(), b.data(), a.size()), 25.0F);
}

TEST(L2SquaredScalar, ZeroDimIsZero) {
  // dim 0 must not read either pointer, so null is a valid argument.
  EXPECT_EQ(l2_squared_scalar(nullptr, nullptr, 0), 0.0F);
}

TEST(L2SquaredScalar, OneDim) {
  const float a = -2.0F;
  const float b = 3.0F;
  EXPECT_EQ(l2_squared_scalar(&a, &b, 1), 25.0F);
}

TEST(L2SquaredScalar, NonMultipleOfEightDim) {
  // dim 13 exercises the tail that SIMD kernels handle separately (8 + 5).
  constexpr std::size_t kDim = 13;
  std::vector<float> a(kDim);
  std::vector<float> b(kDim);
  float expected = 0.0F;
  for (std::size_t i = 0; i < kDim; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i + 1);
    const float diff = a[i] - b[i];  // -(i + 1)
    expected += diff * diff;
  }
  // sum_{j=1}^{13} j^2 = 13 * 14 * 27 / 6 = 819
  EXPECT_EQ(expected, 819.0F);
  EXPECT_EQ(l2_squared_scalar(a.data(), b.data(), kDim), 819.0F);
}

TEST(L2SquaredScalar, IdenticalVectorsAreZero) {
  const std::vector<float> a{0.5F, -1.25F, 3.0F, 7.75F, -0.125F};
  EXPECT_EQ(l2_squared_scalar(a.data(), a.data(), a.size()), 0.0F);
}

TEST(InnerProductDistanceScalar, HandComputed) {
  const std::array<float, 3> a{1.0F, 2.0F, 3.0F};
  const std::array<float, 3> b{4.0F, -5.0F, 6.0F};
  // dot = 4 - 10 + 18 = 12, distance = 1 - 12 = -11
  EXPECT_EQ(inner_product_distance_scalar(a.data(), b.data(), a.size()), -11.0F);
}

TEST(InnerProductDistanceScalar, ZeroDimIsOne) {
  // An empty dot product is 0, so the distance is 1 - 0.
  EXPECT_EQ(inner_product_distance_scalar(nullptr, nullptr, 0), 1.0F);
}

TEST(InnerProductDistanceScalar, OneDim) {
  const float a = 3.0F;
  const float b = 0.5F;
  EXPECT_EQ(inner_product_distance_scalar(&a, &b, 1), -0.5F);
}

TEST(InnerProductDistanceScalar, NonMultipleOfEightDim) {
  constexpr std::size_t kDim = 13;
  std::vector<float> a(kDim, 1.0F);
  std::vector<float> b(kDim);
  for (std::size_t i = 0; i < kDim; ++i) {
    b[i] = static_cast<float>(i + 1);
  }
  // dot = 1 + 2 + ... + 13 = 91
  EXPECT_EQ(inner_product_distance_scalar(a.data(), b.data(), kDim), 1.0F - 91.0F);
}

TEST(InnerProductDistanceScalar, IdenticalUnitVectorsAreZero) {
  const std::array<float, 4> a{0.5F, 0.5F, 0.5F, 0.5F};  // norm 1
  EXPECT_EQ(inner_product_distance_scalar(a.data(), a.data(), a.size()), 0.0F);
}

TEST(Normalize, ScalesToUnitNorm) {
  std::array<float, 2> v{3.0F, 4.0F};
  normalize(v.data(), v.size());
  EXPECT_FLOAT_EQ(v[0], 0.6F);
  EXPECT_FLOAT_EQ(v[1], 0.8F);
}

TEST(Normalize, NonMultipleOfEightDimHasUnitNorm) {
  constexpr std::size_t kDim = 13;
  std::vector<float> v(kDim);
  for (std::size_t i = 0; i < kDim; ++i) {
    v[i] = static_cast<float>(i) - 6.5F;
  }
  normalize(v.data(), kDim);
  // 1 - dot(v, v) should be ~0 for a unit vector.
  EXPECT_NEAR(inner_product_distance_scalar(v.data(), v.data(), kDim), 0.0F, 1e-6F);
}

TEST(Normalize, ZeroVectorIsUnchanged) {
  std::array<float, 3> v{0.0F, 0.0F, 0.0F};
  normalize(v.data(), v.size());
  for (const float x : v) {
    EXPECT_EQ(x, 0.0F);  // and in particular not NaN
  }
}

TEST(Normalize, ZeroDimIsNoOp) { normalize(nullptr, 0); }

TEST(Dispatch, ResolvesWorkingKernelForEveryMetric) {
  const std::array<float, 2> a{0.6F, 0.8F};  // unit norm
  const std::array<float, 2> b{1.0F, 0.0F};

  const DistanceFn l2 = resolve_distance(Metric::L2);
  ASSERT_NE(l2, nullptr);
  EXPECT_FLOAT_EQ(l2(a.data(), b.data(), a.size()),
                  l2_squared_scalar(a.data(), b.data(), a.size()));

  for (const Metric m : {Metric::InnerProduct, Metric::Cosine}) {
    SCOPED_TRACE(to_string(m));
    const DistanceFn fn = resolve_distance(m);
    ASSERT_NE(fn, nullptr);
    EXPECT_FLOAT_EQ(fn(a.data(), b.data(), a.size()), 1.0F - 0.6F);
  }
}

TEST(Dispatch, ActiveKernelNameIsSet) {
  const char* name = active_kernel_name();
  ASSERT_NE(name, nullptr);
  EXPECT_FALSE(std::string_view(name).empty());
}

}  // namespace
}  // namespace annex
