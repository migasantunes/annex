#include "annex/flat_index.hpp"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace annex {
namespace {

void check_dim(std::size_t got, std::size_t expected, const char* what) {
  if (got != expected) {
    throw std::invalid_argument(std::string(what) + " has dimension " + std::to_string(got) +
                                ", index expects " + std::to_string(expected));
  }
}

}  // namespace

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric), distance_(resolve_distance(metric)) {
  if (dim == 0) {
    throw std::invalid_argument("FlatIndex: dim must be > 0");
  }
}

void FlatIndex::add(Label label, std::span<const float> vector) {
  check_dim(vector.size(), dim_, "FlatIndex::add: vector");

  // try_emplace both detects a duplicate and reserves the label in one hash
  // lookup. If it throws (allocation), nothing has been modified yet.
  const std::size_t slot = labels_.size();
  if (!slot_of_.try_emplace(label, slot).second) {
    throw std::invalid_argument("FlatIndex::add: label " + std::to_string(label) +
                                " is already present");
  }

  // If growing either vector throws, undo the partial insert so a failed add()
  // leaves the index exactly as it was (strong exception guarantee). Shrinking
  // a vector of trivial types never throws, so the rollback itself is safe.
  try {
    labels_.push_back(label);
    data_.insert(data_.end(), vector.begin(), vector.end());
  } catch (...) {
    slot_of_.erase(label);
    labels_.resize(slot);
    data_.resize(slot * dim_);
    throw;
  }

  if (metric_ == Metric::Cosine) {
    // Normalising the stored copy once on insert means every search pays only
    // for an inner product.
    normalize(data_.data() + slot * dim_, dim_);
  }
}

std::vector<Neighbor> FlatIndex::search(std::span<const float> query, std::size_t k) const {
  check_dim(query.size(), dim_, "FlatIndex::search: query");

  const std::size_t n = labels_.size();
  k = std::min(k, n);
  if (k == 0) {
    return {};
  }

  const float* q = query.data();
  std::vector<float> normalized_query;
  if (metric_ == Metric::Cosine) {
    // The caller's query is const and must not be modified; normalise a copy.
    normalized_query.assign(query.begin(), query.end());
    normalize(normalized_query.data(), dim_);
    q = normalized_query.data();
  }

  // Bounded max-heap of the k best seen so far. Its top is the *worst* of the
  // current best, so each new candidate is a single comparison against it.
  // This is O(n log k) time and O(k) memory, versus O(n log n) and O(n) for
  // computing every distance and sorting. Ordering by Neighbor::operator<
  // (distance, then label) makes the kept set deterministic under ties.
  std::priority_queue<Neighbor> heap;

  for (std::size_t slot = 0; slot < n; ++slot) {
    const Neighbor candidate{labels_[slot], distance_(q, data_.data() + slot * dim_, dim_)};
    if (heap.size() < k) {
      heap.push(candidate);
    } else if (candidate < heap.top()) {
      heap.pop();
      heap.push(candidate);
    }
  }

  // Popping a max-heap yields worst-first, so fill the result from the back.
  std::vector<Neighbor> result(heap.size());
  for (auto it = result.rbegin(); it != result.rend(); ++it) {
    *it = heap.top();
    heap.pop();
  }
  return result;
}

void FlatIndex::reserve(std::size_t n) {
  data_.reserve(n * dim_);
  labels_.reserve(n);
  slot_of_.reserve(n);
}

}  // namespace annex
