// The interface every index implements.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "annex/types.hpp"

namespace annex {

/// A collection of fixed-dimension vectors that can be searched by nearest
/// neighbour.
///
/// Contract shared by all implementations:
///   * add() throws std::invalid_argument if vector.size() != dim() or if
///     `label` is already present.
///   * search() throws std::invalid_argument if query.size() != dim().
///   * search() returns min(k, size()) hits sorted nearest-first (see
///     Neighbor::operator<). k == 0 or an empty index returns an empty vector.
///   * Vectors are copied on add(); the caller keeps ownership of its buffer.
///
/// Thread-safety is implementation-defined and documented per class.
class Index {
 public:
  Index() = default;
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  Index(Index&&) = default;
  Index& operator=(Index&&) = default;
  virtual ~Index() = default;

  virtual void add(Label label, std::span<const float> vector) = 0;

  [[nodiscard]] virtual std::vector<Neighbor> search(std::span<const float> query,
                                                     std::size_t k) const = 0;

  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  [[nodiscard]] virtual std::size_t dim() const noexcept = 0;
  [[nodiscard]] virtual Metric metric() const noexcept = 0;
};

}  // namespace annex
