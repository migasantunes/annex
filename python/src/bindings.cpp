// pybind11 bindings: the annex._core extension module.
//
// This file is deliberately thin. It turns NumPy arrays into spans for the
// C++ core and results back into NumPy arrays, validates arguments that only
// make sense on the Python side (shapes, label arrays), and releases the GIL
// around the loops that do real work. All search logic stays in the core.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "annex/distance.hpp"
#include "annex/flat_index.hpp"
#include "annex/types.hpp"

namespace py = pybind11;

namespace {

// c_style | forcecast: a C-contiguous float32 array is passed through without
// a copy; anything else (float64, Fortran order, strided views, lists) is
// converted into a fresh C-contiguous float32 array before we see it. Either
// way the core gets one flat row-major buffer.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

annex::Metric parse_metric(std::string_view name) {
  if (name == "l2") return annex::Metric::L2;
  if (name == "ip") return annex::Metric::InnerProduct;
  if (name == "cosine") return annex::Metric::Cosine;
  throw py::value_error("metric must be 'l2', 'ip' or 'cosine', got '" + std::string(name) + "'");
}

std::string shape_string(const py::array& a) {
  // Formatted like a Python tuple, (3,) and (2, 5), to match NumPy's messages.
  std::string s = "(";
  for (py::ssize_t i = 0; i < a.ndim(); ++i) {
    if (i > 0) s += ", ";
    s += std::to_string(a.shape(i));
  }
  if (a.ndim() == 1) s += ",";
  return s + ")";
}

// A batch of row vectors viewed as a matrix. Accepting a 1-D array of length
// dim as a single row saves callers a reshape for the common one-query case.
struct Rows {
  const float* data;
  std::size_t count;
};

Rows as_rows(const FloatArray& a, std::size_t dim, const char* what) {
  const bool ok = (a.ndim() == 1 && static_cast<std::size_t>(a.shape(0)) == dim) ||
                  (a.ndim() == 2 && static_cast<std::size_t>(a.shape(1)) == dim);
  if (!ok) {
    throw py::value_error(std::string(what) + " must have shape (n, " + std::to_string(dim) +
                          ") or (" + std::to_string(dim) + ",), got " + shape_string(a));
  }
  const std::size_t count = a.ndim() == 1 ? 1 : static_cast<std::size_t>(a.shape(0));
  return {a.data(), count};
}

// Labels are converted by hand rather than with forcecast because forcecast
// would silently wrap a negative int64 into a huge uint64, and NumPy creates
// int64 arrays by default (np.array([0, 1, 2])), so that mistake is easy.
std::vector<annex::Label> parse_labels(const py::handle& obj, std::size_t n) {
  py::array arr = py::array::ensure(obj);
  if (!arr) {
    throw py::value_error("labels must be a 1-D array of non-negative integers");
  }
  if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != n) {
    throw py::value_error("labels must have shape (" + std::to_string(n) +
                          ",) to match the number of vectors, got " + shape_string(arr));
  }

  std::vector<annex::Label> labels(n);
  if (n == 0) {
    // An empty list becomes a float64 array in NumPy; there is nothing to check.
    return labels;
  }

  const char kind = arr.dtype().kind();
  if (kind == 'u') {
    auto u = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    std::copy_n(u.data(), n, labels.begin());
  } else if (kind == 'i') {
    auto s = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    const std::int64_t* p = s.data();
    for (std::size_t i = 0; i < n; ++i) {
      if (p[i] < 0) {
        throw py::value_error("labels must be non-negative, got " + std::to_string(p[i]) +
                              " at position " + std::to_string(i));
      }
      labels[i] = static_cast<annex::Label>(p[i]);
    }
  } else {
    throw py::value_error("labels must have an integer dtype, got " +
                          py::str(arr.dtype()).cast<std::string>());
  }

  // Catch duplicates inside the batch before anything is added, so this common
  // mistake leaves the index untouched. Sorting a copy is O(n log n) and needs
  // no hash set.
  std::vector<annex::Label> sorted = labels;
  std::sort(sorted.begin(), sorted.end());
  const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
  if (dup != sorted.end()) {
    throw py::value_error("labels contains duplicate label " + std::to_string(*dup));
  }
  return labels;
}

// FlatIndex allows concurrent search() calls but not add() alongside anything
// else. While the GIL was held that was automatic; releasing it means two
// Python threads can now be inside the core at once, so the bindings take over
// that job with a reader-writer lock: search shares it, add takes it alone.
//
// Lock order matters: nothing may wait for the GIL while holding the mutex. A
// thread that held the mutex and then waited for the GIL could deadlock with a
// thread that held the GIL and waited for the mutex. So add() and search()
// release the GIL first, take the mutex second, and (by declaration order)
// drop the mutex before re-acquiring the GIL. size() takes the mutex with the
// GIL held, which is safe for the same reason: whoever holds the mutex can
// always finish and release it without needing the GIL.
class PyFlatIndex {
 public:
  PyFlatIndex(std::int64_t dim, const std::string& metric)
      : index_(checked_dim(dim), parse_metric(metric)) {}

  void add(const FloatArray& vectors, const py::object& labels_obj) {
    const Rows rows = as_rows(vectors, index_.dim(), "vectors");
    const bool explicit_labels = !labels_obj.is_none();
    std::vector<annex::Label> labels;
    if (explicit_labels) {
      labels = parse_labels(labels_obj, rows.count);
    }
    if (rows.count == 0) {
      return;
    }

    // From here on only C++ data is touched: rows.data points into `vectors`,
    // which this frame keeps alive, and `labels` is a std::vector.
    py::gil_scoped_release no_gil;
    std::unique_lock lock(mutex_);

    if (!explicit_labels) {
      // Read size() under the exclusive lock so two concurrent default-label
      // add() calls cannot both pick the same starting label.
      labels.resize(rows.count);
      std::iota(labels.begin(), labels.end(), static_cast<annex::Label>(index_.size()));
    }

    const std::size_t dim = index_.dim();
    index_.reserve(index_.size() + rows.count);
    for (std::size_t i = 0; i < rows.count; ++i) {
      // Can still throw if a label is already in the index. Rows before it stay
      // added: undoing them would need a remove() the core does not have.
      index_.add(labels[i], std::span<const float>(rows.data + i * dim, dim));
    }
  }

  py::tuple search(const FloatArray& queries, std::int64_t k_arg) const {
    const Rows rows = as_rows(queries, index_.dim(), "queries");
    if (k_arg < 0) {
      throw py::value_error("k must be non-negative, got " + std::to_string(k_arg));
    }
    const auto k = static_cast<std::size_t>(k_arg);
    // The core would quietly return fewer than k hits; we refuse instead so the
    // output is always a rectangular (nq, k) array. size() only ever grows, so
    // the check stays true even if another thread adds before we search.
    const std::size_t n = size();
    if (k > n) {
      throw py::value_error("k (" + std::to_string(k) + ") must be <= len(index) (" +
                            std::to_string(n) + ")");
    }

    // Outputs are allocated while we still hold the GIL; afterwards only their
    // raw buffers are written, which needs no GIL because nobody else can see
    // these arrays yet.
    const auto nq = static_cast<py::ssize_t>(rows.count);
    const auto kk = static_cast<py::ssize_t>(k);
    py::array_t<annex::Label> labels({nq, kk});
    py::array_t<float> distances({nq, kk});
    annex::Label* out_labels = labels.mutable_data();
    float* out_distances = distances.mutable_data();

    {
      py::gil_scoped_release no_gil;
      std::shared_lock lock(mutex_);
      const std::size_t dim = index_.dim();
      for (std::size_t q = 0; q < rows.count; ++q) {
        const auto hits = index_.search(std::span<const float>(rows.data + q * dim, dim), k);
        for (std::size_t j = 0; j < k; ++j) {
          out_labels[q * k + j] = hits[j].label;
          out_distances[q * k + j] = hits[j].distance;
        }
      }
    }
    return py::make_tuple(std::move(labels), std::move(distances));
  }

  std::size_t size() const {
    std::shared_lock lock(mutex_);
    return index_.size();
  }

  std::size_t dim() const noexcept { return index_.dim(); }

  std::string metric() const { return std::string(annex::to_string(index_.metric())); }

 private:
  // Taking a signed integer lets us raise ValueError for dim < 0; letting
  // pybind11 convert to size_t would raise a less helpful TypeError.
  static std::size_t checked_dim(std::int64_t dim) {
    if (dim <= 0) {
      throw py::value_error("dim must be > 0, got " + std::to_string(dim));
    }
    return static_cast<std::size_t>(dim);
  }

  annex::FlatIndex index_;
  mutable std::shared_mutex mutex_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = R"doc(
Compiled core of annex, a vector search engine written in C++20.

Import the public API from the ``annex`` package rather than from here.
)doc";

  m.attr("__version__") = ANNEX_VERSION;

  m.def(
      "active_kernel_name", [] { return std::string(annex::active_kernel_name()); },
      R"doc(
Name of the distance-kernel instruction set chosen for this CPU, e.g. ``"scalar"``.

Benchmark results record it so numbers from different machines can be compared.
)doc");

  py::class_<PyFlatIndex>(m, "FlatIndex", R"doc(
Exact brute-force nearest-neighbour index.

Every query is compared with every stored vector, so results are always exact.
It is the correctness reference for the approximate indexes and the
recall = 1.0 baseline in benchmarks.

Distances follow one convention: smaller is more similar.

* ``"l2"``: squared Euclidean distance.
* ``"ip"``: ``1 - dot(a, b)``.
* ``"cosine"``: ``1 - cos(a, b)``. Vectors are normalised when added and when
  queried; a zero vector is left as is.

Thread safety: ``search`` may be called from several threads at once and runs
without the GIL. ``add`` excludes every other call on the same index.
)doc")
      .def(py::init<std::int64_t, const std::string&>(), py::arg("dim"), py::arg("metric") = "l2",
           R"doc(
Create an empty index.

Args:
    dim: Dimension of every vector; must be > 0.
    metric: ``"l2"``, ``"ip"`` or ``"cosine"``.

Raises:
    ValueError: If ``dim <= 0`` or the metric is unknown.
)doc")
      .def("add", &PyFlatIndex::add, py::arg("vectors"), py::arg("labels") = py::none(),
           R"doc(
Add vectors to the index. The index keeps its own copy.

Args:
    vectors: Array of shape ``(n, dim)``, or ``(dim,)`` for a single vector.
        C-contiguous float32 is used without a copy; other dtypes and layouts
        are converted first.
    labels: Optional 1-D integer array of ``n`` non-negative, distinct labels.
        If omitted, labels are ``len(self), ..., len(self) + n - 1``.

Raises:
    ValueError: If the shape does not match ``dim``, ``labels`` has the wrong
        length or dtype, contains a negative value or a duplicate, or a label
        is already in the index.

All of these are checked before anything is added except the last: a label
already in the index is only found when that row is reached, and the rows
before it remain added.
)doc")
      .def("search", &PyFlatIndex::search, py::arg("queries"), py::arg("k"),
           R"doc(
Find the ``k`` nearest stored vectors to each query.

Args:
    queries: Array of shape ``(nq, dim)``, or ``(dim,)`` for a single query,
        which is treated as ``nq = 1``. Converted like ``add``'s input.
    k: Number of neighbours per query; ``0 <= k <= len(self)``.

Returns:
    ``(labels, distances)``: arrays of shape ``(nq, k)`` with dtypes uint64
    and float32. Each row is sorted nearest first; ties are broken by the
    smaller label.

Raises:
    ValueError: If the shape does not match ``dim`` or ``k`` is out of range.
)doc")
      .def("__len__", &PyFlatIndex::size, "Number of vectors stored.")
      .def_property_readonly("dim", &PyFlatIndex::dim, "Dimension of every vector.")
      .def_property_readonly("metric", &PyFlatIndex::metric,
                             "Metric name: ``'l2'``, ``'ip'`` or ``'cosine'``.")
      .def("__repr__", [](const PyFlatIndex& self) {
        return "annex.FlatIndex(dim=" + std::to_string(self.dim()) + ", metric='" + self.metric() +
               "', size=" + std::to_string(self.size()) + ")";
      });
}
