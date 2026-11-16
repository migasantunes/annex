#include "annex/flat_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "annex/distance.hpp"
#include "annex/types.hpp"

namespace annex {
namespace {

// Labels only, in result order: makes EXPECT_EQ failures easy to read.
std::vector<Label> labels_of(const std::vector<Neighbor>& hits) {
  std::vector<Label> out;
  out.reserve(hits.size());
  for (const Neighbor& h : hits) {
    out.push_back(h.label);
  }
  return out;
}

// --- Contract -----------------------------------------------------------------

TEST(FlatIndex, ZeroDimThrows) { EXPECT_THROW(FlatIndex(0, Metric::L2), std::invalid_argument); }

TEST(FlatIndex, ReportsConstructionParameters) {
  const FlatIndex index(4, Metric::Cosine);
  EXPECT_EQ(index.dim(), 4U);
  EXPECT_EQ(index.metric(), Metric::Cosine);
  EXPECT_EQ(index.size(), 0U);
}

TEST(FlatIndex, AddWithWrongDimThrows) {
  FlatIndex index(3, Metric::L2);
  const std::vector<float> short_vec{1.0F, 2.0F};
  const std::vector<float> long_vec{1.0F, 2.0F, 3.0F, 4.0F};
  EXPECT_THROW(index.add(1, short_vec), std::invalid_argument);
  EXPECT_THROW(index.add(1, long_vec), std::invalid_argument);
  EXPECT_EQ(index.size(), 0U);
}

TEST(FlatIndex, DuplicateLabelThrowsAndLeavesIndexUnchanged) {
  FlatIndex index(2, Metric::L2);
  index.add(7, std::vector<float>{0.0F, 0.0F});
  EXPECT_THROW(index.add(7, std::vector<float>{5.0F, 5.0F}), std::invalid_argument);
  EXPECT_EQ(index.size(), 1U);

  // The original vector must still be the one stored under label 7.
  const auto hits = index.search(std::vector<float>{0.0F, 0.0F}, 1);
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_EQ(hits[0], (Neighbor{7, 0.0F}));
}

TEST(FlatIndex, SearchWithWrongDimThrows) {
  FlatIndex index(3, Metric::L2);
  index.add(1, std::vector<float>{1.0F, 2.0F, 3.0F});
  EXPECT_THROW(std::ignore = index.search(std::vector<float>{1.0F, 2.0F}, 1),
               std::invalid_argument);
}

TEST(FlatIndex, SearchWithWrongDimThrowsEvenWhenEmptyOrKZero) {
  // Validation comes before the early returns, so misuse is never silent.
  FlatIndex index(3, Metric::L2);
  EXPECT_THROW(std::ignore = index.search(std::vector<float>{1.0F}, 5), std::invalid_argument);
  index.add(1, std::vector<float>{1.0F, 2.0F, 3.0F});
  EXPECT_THROW(std::ignore = index.search(std::vector<float>{1.0F}, 0), std::invalid_argument);
}

TEST(FlatIndex, KZeroReturnsEmpty) {
  FlatIndex index(2, Metric::L2);
  index.add(1, std::vector<float>{1.0F, 1.0F});
  EXPECT_TRUE(index.search(std::vector<float>{0.0F, 0.0F}, 0).empty());
}

TEST(FlatIndex, EmptyIndexReturnsEmpty) {
  const FlatIndex index(2, Metric::L2);
  EXPECT_TRUE(index.search(std::vector<float>{0.0F, 0.0F}, 10).empty());
}

TEST(FlatIndex, KLargerThanSizeReturnsAll) {
  FlatIndex index(1, Metric::L2);
  for (Label l = 0; l < 3; ++l) {
    index.add(l, std::vector<float>{static_cast<float>(l)});
  }
  EXPECT_EQ(index.search(std::vector<float>{0.0F}, 100).size(), 3U);
}

TEST(FlatIndex, ResultsSortedNearestFirst) {
  FlatIndex index(1, Metric::L2);
  // Inserted out of order so sorting can't come from insertion order.
  const std::vector<std::pair<Label, float>> points{
      {10, 4.0F}, {11, -1.0F}, {12, 2.5F}, {13, 0.5F}, {14, -3.0F}};
  for (const auto& [label, x] : points) {
    index.add(label, std::vector<float>{x});
  }
  const auto hits = index.search(std::vector<float>{0.0F}, 5);
  // distances: 13 -> 0.25, 11 -> 1, 12 -> 6.25, 14 -> 9, 10 -> 16
  EXPECT_EQ(labels_of(hits), (std::vector<Label>{13, 11, 12, 14, 10}));
  EXPECT_TRUE(std::is_sorted(hits.begin(), hits.end()));
}

TEST(FlatIndex, TiesBrokenByAscendingLabel) {
  FlatIndex index(1, Metric::L2);
  // Every point is distance 1 from the query; labels inserted in scrambled order.
  for (const Label l : {Label{42}, Label{3}, Label{17}, Label{8}, Label{25}}) {
    const float x = (l % 2 == 0) ? 1.0F : -1.0F;
    index.add(l, std::vector<float>{x});
  }
  const std::vector<float> query{0.0F};
  EXPECT_EQ(labels_of(index.search(query, 5)), (std::vector<Label>{3, 8, 17, 25, 42}));
  // With k < size, the tie must also decide *which* points are kept.
  EXPECT_EQ(labels_of(index.search(query, 2)), (std::vector<Label>{3, 8}));
}

TEST(FlatIndex, CopiesCallerBuffer) {
  FlatIndex index(2, Metric::L2);
  std::vector<float> buffer{1.0F, 0.0F};
  index.add(1, buffer);
  buffer = {100.0F, 100.0F};  // must not affect what the index stored
  index.add(2, buffer);

  const auto hits = index.search(std::vector<float>{1.0F, 0.0F}, 1);
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_EQ(hits[0], (Neighbor{1, 0.0F}));
}

TEST(FlatIndex, CosineDoesNotModifyCallerBuffers) {
  FlatIndex index(2, Metric::Cosine);
  const std::vector<float> vec{3.0F, 4.0F};
  const std::vector<float> query{6.0F, 8.0F};
  index.add(1, vec);
  std::ignore = index.search(query, 1);
  EXPECT_EQ(vec, (std::vector<float>{3.0F, 4.0F}));
  EXPECT_EQ(query, (std::vector<float>{6.0F, 8.0F}));
}

TEST(FlatIndex, ReserveDoesNotChangeSize) {
  FlatIndex index(4, Metric::L2);
  index.reserve(1000);
  EXPECT_EQ(index.size(), 0U);
}

// --- Per-metric correctness on hand-checkable data -----------------------------

TEST(FlatIndex, L2NeighboursAndDistances) {
  FlatIndex index(2, Metric::L2);
  index.add(1, std::vector<float>{0.0F, 0.0F});
  index.add(2, std::vector<float>{3.0F, 4.0F});
  index.add(3, std::vector<float>{1.0F, 1.0F});
  index.add(4, std::vector<float>{-2.0F, 0.0F});

  const auto hits = index.search(std::vector<float>{1.0F, 0.0F}, 3);
  // squared distances from (1, 0): 1 -> 1, 2 -> 4 + 16 = 20, 3 -> 1, 4 -> 9
  const std::vector<Neighbor> expected{{1, 1.0F}, {3, 1.0F}, {4, 9.0F}};
  EXPECT_EQ(hits, expected);
}

TEST(FlatIndex, InnerProductNeighboursAndDistances) {
  FlatIndex index(2, Metric::InnerProduct);
  // Not normalised: inner product rewards magnitude as well as direction.
  index.add(1, std::vector<float>{1.0F, 0.0F});
  index.add(2, std::vector<float>{0.0F, 1.0F});
  index.add(3, std::vector<float>{2.0F, 2.0F});
  index.add(4, std::vector<float>{-1.0F, 0.0F});

  const auto hits = index.search(std::vector<float>{1.0F, 0.5F}, 4);
  // dot with (1, 0.5): 1 -> 1, 2 -> 0.5, 3 -> 3, 4 -> -1; distance = 1 - dot
  const std::vector<Neighbor> expected{{3, -2.0F}, {1, 0.0F}, {2, 0.5F}, {4, 2.0F}};
  EXPECT_EQ(hits, expected);
}

TEST(FlatIndex, CosineNeighboursAndDistancesWithUnnormalisedInputs) {
  FlatIndex index(2, Metric::Cosine);
  index.add(1, std::vector<float>{10.0F, 0.0F});  // angle 0 to the x-axis
  index.add(2, std::vector<float>{0.0F, 0.5F});   // 90 degrees
  index.add(3, std::vector<float>{3.0F, 3.0F});   // 45 degrees
  index.add(4, std::vector<float>{-7.0F, 0.0F});  // 180 degrees

  // Query along the x-axis with an arbitrary magnitude: magnitude must not matter.
  const auto hits = index.search(std::vector<float>{0.25F, 0.0F}, 4);
  ASSERT_EQ(hits.size(), 4U);
  EXPECT_EQ(labels_of(hits), (std::vector<Label>{1, 3, 2, 4}));
  // 1 - cos(theta) for theta = 0, 45, 90, 180 degrees.
  EXPECT_NEAR(hits[0].distance, 0.0F, 1e-6F);
  EXPECT_NEAR(hits[1].distance, 1.0F - std::sqrt(0.5F), 1e-6F);
  EXPECT_NEAR(hits[2].distance, 1.0F, 1e-6F);
  EXPECT_NEAR(hits[3].distance, 2.0F, 1e-6F);
}

// --- Randomised property test against a naive reference ---------------------------

// Independent of FlatIndex: computes every distance with its own loops, fully
// sorts, and truncates. Deliberately the most obvious possible implementation.
std::vector<Neighbor> naive_search(const std::vector<std::vector<float>>& data,
                                   const std::vector<float>& query, std::size_t k, Metric metric) {
  auto prepared = [metric](std::vector<float> v) {
    if (metric == Metric::Cosine) {
      normalize(v.data(), v.size());
    }
    return v;
  };
  const std::vector<float> q = prepared(query);

  std::vector<Neighbor> all;
  all.reserve(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    const std::vector<float> v = prepared(data[i]);
    float d = 0.0F;
    if (metric == Metric::L2) {
      for (std::size_t j = 0; j < v.size(); ++j) {
        d += (v[j] - q[j]) * (v[j] - q[j]);
      }
    } else {
      float dot = 0.0F;
      for (std::size_t j = 0; j < v.size(); ++j) {
        dot += v[j] * q[j];
      }
      d = 1.0F - dot;
    }
    all.push_back({static_cast<Label>(i), d});
  }
  std::sort(all.begin(), all.end());
  all.resize(std::min(k, all.size()));
  return all;
}

struct PropertyCase {
  std::size_t n;
  std::size_t dim;
  std::size_t k;
};

// Lets GoogleTest print the case readably when a test fails.
void PrintTo(const PropertyCase& c, std::ostream* os) {
  *os << "{n=" << c.n << ", dim=" << c.dim << ", k=" << c.k << "}";
}

class FlatIndexProperty : public ::testing::TestWithParam<std::tuple<Metric, PropertyCase>> {};

TEST_P(FlatIndexProperty, MatchesNaiveReference) {
  const auto [metric, c] = GetParam();
  // Fixed seed: a failure here is reproducible, not flaky.
  std::mt19937 rng(12345U + static_cast<unsigned>(c.n * 31 + c.dim * 7 + c.k));
  // Integer-valued coordinates in a small range produce many exact distance
  // ties, which is what actually exercises the label tie-break. They also keep
  // L2 and inner-product arithmetic exact, so EXPECT_EQ on float distances does
  // not depend on summation order. Cosine is not exact after normalisation: it
  // matches today only because both sides use the same scalar loop order.
  std::uniform_int_distribution<int> coord(-3, 3);

  FlatIndex index(c.dim, metric);
  std::vector<std::vector<float>> data(c.n, std::vector<float>(c.dim));
  for (std::size_t i = 0; i < c.n; ++i) {
    for (float& x : data[i]) {
      x = static_cast<float>(coord(rng));
    }
    index.add(static_cast<Label>(i), data[i]);
  }

  constexpr int kQueries = 10;
  for (int qi = 0; qi < kQueries; ++qi) {
    std::vector<float> query(c.dim);
    for (float& x : query) {
      x = static_cast<float>(coord(rng));
    }
    SCOPED_TRACE(::testing::Message() << "query " << qi);
    EXPECT_EQ(index.search(query, c.k), naive_search(data, query, c.k, metric));
  }
}

// Readable test names, e.g. "cosine_n50_dim13_k10". A named function rather
// than a lambda because a structured binding's comma would split the macro.
std::string property_case_name(const ::testing::TestParamInfo<FlatIndexProperty::ParamType>& info) {
  const Metric metric = std::get<0>(info.param);
  const PropertyCase& c = std::get<1>(info.param);
  return std::string(to_string(metric)) + "_n" + std::to_string(c.n) + "_dim" +
         std::to_string(c.dim) + "_k" + std::to_string(c.k);
}

INSTANTIATE_TEST_SUITE_P(
    RandomData, FlatIndexProperty,
    ::testing::Combine(::testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine),
                       ::testing::Values(PropertyCase{1, 1, 1}, PropertyCase{10, 3, 5},
                                         PropertyCase{50, 13, 10}, PropertyCase{200, 32, 1},
                                         PropertyCase{500, 64, 20}, PropertyCase{100, 8, 150})),
    property_case_name);

}  // namespace
}  // namespace annex
