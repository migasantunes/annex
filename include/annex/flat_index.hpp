// Exact brute-force index.
#pragma once

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

#include "annex/distance.hpp"
#include "annex/index.hpp"
#include "annex/types.hpp"

namespace annex {

/// Compares the query against every stored vector. O(n * dim) per query and
/// always exactly right.
///
/// It exists for two reasons: it is the correctness oracle that every
/// approximate index is tested against, and it is the "recall = 1.0" baseline
/// on the benchmark curves.
///
/// Thread-safety: concurrent search() calls are safe. add() must not run
/// concurrently with anything else.
class FlatIndex final : public Index {
 public:
  /// Throws std::invalid_argument if dim == 0.
  FlatIndex(std::size_t dim, Metric metric);

  void add(Label label, std::span<const float> vector) override;

  [[nodiscard]] std::vector<Neighbor> search(std::span<const float> query,
                                             std::size_t k) const override;

  [[nodiscard]] std::size_t size() const noexcept override { return labels_.size(); }
  [[nodiscard]] std::size_t dim() const noexcept override { return dim_; }
  [[nodiscard]] Metric metric() const noexcept override { return metric_; }

  /// Pre-allocate room for `n` vectors in total.
  void reserve(std::size_t n);

 private:
  std::size_t dim_;
  Metric metric_;
  DistanceFn distance_;
  std::vector<float> data_;                         // size() * dim_, row-major, contiguous
  std::vector<Label> labels_;                       // labels_[slot] is the label of row `slot`
  std::unordered_map<Label, std::size_t> slot_of_;  // duplicate-label detection
};

}  // namespace annex
