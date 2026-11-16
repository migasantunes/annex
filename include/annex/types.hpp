// Core vocabulary types shared by every index implementation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace annex {

/// Caller-chosen identifier for a stored vector. Opaque to the engine.
using Label = std::uint64_t;

/// How two vectors are compared.
///
/// Every metric is exposed as a *distance*: smaller means more similar.
///   L2            squared Euclidean distance (no sqrt; ordering is identical)
///   InnerProduct  1 - dot(a, b)
///   Cosine        1 - cos(a, b); vectors are L2-normalised on the way in,
///                 after which it is computed exactly like InnerProduct
enum class Metric : std::uint8_t {
  L2 = 0,
  InnerProduct = 1,
  Cosine = 2,
};

[[nodiscard]] constexpr std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

/// One search hit.
struct Neighbor {
  Label label;
  float distance;

  /// Total order used everywhere results are ranked: nearest first, ties
  /// broken by ascending label so results are deterministic.
  [[nodiscard]] friend constexpr bool operator<(const Neighbor& a, const Neighbor& b) noexcept {
    return a.distance < b.distance || (a.distance == b.distance && a.label < b.label);
  }
  [[nodiscard]] friend constexpr bool operator==(const Neighbor& a, const Neighbor& b) noexcept {
    return a.label == b.label && a.distance == b.distance;
  }
};

}  // namespace annex
