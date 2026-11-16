"""Tests for the annex.FlatIndex bindings.

The C++ behaviour itself is covered by GoogleTest; these tests check what the
binding layer adds: array conversion, shapes and dtypes, label handling, error
translation, and that results agree with an independent NumPy implementation.
"""

import importlib.metadata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import reference_search

import annex

DIM = 16


def make_index(rng, n=200, dim=DIM, metric="l2"):
    data = rng.standard_normal((n, dim)).astype(np.float32)
    index = annex.FlatIndex(dim=dim, metric=metric)
    index.add(data)
    return index, data


# --- Construction and properties ---------------------------------------------


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_properties(metric):
    index = annex.FlatIndex(dim=8, metric=metric)
    assert index.dim == 8
    assert index.metric == metric
    assert len(index) == 0


def test_metric_defaults_to_l2():
    assert annex.FlatIndex(dim=3).metric == "l2"


@pytest.mark.parametrize("metric", ["L2", "euclidean", "", "dot"])
def test_unknown_metric_raises(metric):
    with pytest.raises(ValueError, match="metric"):
        annex.FlatIndex(dim=8, metric=metric)


@pytest.mark.parametrize("dim", [0, -1])
def test_bad_dim_raises(dim):
    with pytest.raises(ValueError, match="dim"):
        annex.FlatIndex(dim=dim)


def test_repr(rng):
    index, _ = make_index(rng, n=1000, dim=128)
    assert repr(index) == "annex.FlatIndex(dim=128, metric='l2', size=1000)"


def test_version():
    assert annex.__version__ == "0.1.0"
    # The compiled version (from CMake) must match the packaging metadata.
    assert annex.__version__ == importlib.metadata.version("annex")


def test_active_kernel_name():
    name = annex.active_kernel_name()
    assert isinstance(name, str) and name


# --- Output shapes and dtypes ------------------------------------------------


def test_search_output_shapes_and_dtypes(rng):
    index, _ = make_index(rng)
    queries = rng.standard_normal((7, DIM)).astype(np.float32)
    labels, distances = index.search(queries, k=5)
    assert labels.shape == (7, 5) and labels.dtype == np.uint64
    assert distances.shape == (7, 5) and distances.dtype == np.float32
    # Nearest first.
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_k_zero_returns_empty_columns(rng):
    index, data = make_index(rng)
    labels, distances = index.search(data[:4], k=0)
    assert labels.shape == (4, 0) and labels.dtype == np.uint64
    assert distances.shape == (4, 0) and distances.dtype == np.float32


def test_k_zero_on_empty_index():
    labels, distances = annex.FlatIndex(dim=3).search(np.zeros((2, 3), np.float32), k=0)
    assert labels.shape == (2, 0) and distances.shape == (2, 0)


def test_k_equal_to_len_is_allowed(rng):
    index, data = make_index(rng, n=10)
    labels, _ = index.search(data[0], k=10)
    assert sorted(labels[0].tolist()) == list(range(10))


# --- Labels ------------------------------------------------------------------


def test_default_labels_continue_from_len(rng):
    data = rng.standard_normal((10, DIM)).astype(np.float32)
    index = annex.FlatIndex(dim=DIM)
    index.add(data[:6])
    index.add(data[6:])
    assert len(index) == 10
    # Each vector is its own nearest neighbour at distance 0.
    labels, distances = index.search(data, k=1)
    np.testing.assert_array_equal(labels[:, 0], np.arange(10, dtype=np.uint64))
    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-5)


def test_explicit_labels(rng):
    data = rng.standard_normal((5, DIM)).astype(np.float32)
    given = np.array([100, 7, 2**63 + 5, 42, 0], dtype=np.uint64)
    index = annex.FlatIndex(dim=DIM)
    index.add(data, labels=given)
    labels, _ = index.search(data, k=1)
    np.testing.assert_array_equal(labels[:, 0], given)


def test_labels_as_int64_and_list(rng):
    data = rng.standard_normal((3, DIM)).astype(np.float32)
    index = annex.FlatIndex(dim=DIM)
    index.add(data[:2], labels=np.array([10, 11]))  # NumPy's default int64
    index.add(data[2], labels=[12])
    labels, _ = index.search(data, k=1)
    np.testing.assert_array_equal(labels[:, 0], [10, 11, 12])


def test_default_labels_after_explicit(rng):
    # Defaults are positions, not "max label + 1": after adding labels 0 and 5
    # the next default label is len(index) == 2.
    data = rng.standard_normal((3, DIM)).astype(np.float32)
    index = annex.FlatIndex(dim=DIM)
    index.add(data[:2], labels=np.array([0, 5], dtype=np.uint64))
    index.add(data[2])
    labels, _ = index.search(data[2], k=1)
    assert labels[0, 0] == 2


# --- Input conversion --------------------------------------------------------


def test_one_dimensional_input(rng):
    index, data = make_index(rng)
    labels_1d, dist_1d = index.search(data[3], k=4)
    labels_2d, dist_2d = index.search(data[3:4], k=4)
    assert labels_1d.shape == (1, 4)
    np.testing.assert_array_equal(labels_1d, labels_2d)
    np.testing.assert_array_equal(dist_1d, dist_2d)

    single = annex.FlatIndex(dim=DIM)
    single.add(data[0])
    assert len(single) == 1


def _variants(array):
    """Same values as `array` in layouts and dtypes the bindings must convert."""
    wide = np.zeros((array.shape[0], array.shape[1] * 2), dtype=np.float32)
    wide[:, ::2] = array
    return {
        "float64": array.astype(np.float64),
        "fortran": np.asfortranarray(array),
        "strided": wide[:, ::2],
        "list": array.tolist(),
    }


@pytest.mark.parametrize("variant", ["float64", "fortran", "strided", "list"])
def test_converted_queries_match_float32(rng, variant):
    index, _ = make_index(rng)
    queries = rng.standard_normal((9, DIM)).astype(np.float32)
    expected = index.search(queries, k=5)
    other = _variants(queries)[variant]
    if isinstance(other, np.ndarray):
        assert not (other.dtype == np.float32 and other.flags.c_contiguous)
    got = index.search(other, k=5)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@pytest.mark.parametrize("variant", ["float64", "fortran", "strided", "list"])
def test_converted_vectors_match_float32(rng, variant):
    data = rng.standard_normal((50, DIM)).astype(np.float32)
    queries = rng.standard_normal((5, DIM)).astype(np.float32)
    reference = annex.FlatIndex(dim=DIM)
    reference.add(data)
    converted = annex.FlatIndex(dim=DIM)
    converted.add(_variants(data)[variant])
    for got, expected in zip(converted.search(queries, k=5), reference.search(queries, k=5)):
        np.testing.assert_array_equal(got, expected)


# --- Errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(5, DIM + 1), (5, DIM - 1), (DIM + 1,), (2, 2, DIM), ()], ids=str
)
def test_wrong_shape_raises(rng, shape):
    index, _ = make_index(rng)
    bad = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        index.add(bad)
    with pytest.raises(ValueError, match="shape"):
        index.search(bad, k=1)
    assert len(index) == 200


def test_labels_wrong_length_raises(rng):
    index = annex.FlatIndex(dim=DIM)
    data = rng.standard_normal((4, DIM)).astype(np.float32)
    with pytest.raises(ValueError, match="labels"):
        index.add(data, labels=np.arange(3, dtype=np.uint64))
    with pytest.raises(ValueError, match="labels"):
        index.add(data, labels=np.arange(8, dtype=np.uint64).reshape(4, 2))
    assert len(index) == 0


@pytest.mark.parametrize(
    "labels",
    [np.array([0, -1, 2, 3]), np.array([0.0, 1.0, 2.0, 3.0]), np.array(["a", "b", "c", "d"])],
    ids=["negative", "float", "str"],
)
def test_bad_label_values_raise(rng, labels):
    index = annex.FlatIndex(dim=DIM)
    with pytest.raises(ValueError, match="labels"):
        index.add(rng.standard_normal((4, DIM)), labels=labels)
    assert len(index) == 0


def test_duplicate_labels_within_batch_add_nothing(rng):
    index = annex.FlatIndex(dim=DIM)
    data = rng.standard_normal((4, DIM)).astype(np.float32)
    with pytest.raises(ValueError, match="duplicate"):
        index.add(data, labels=np.array([1, 2, 3, 2], dtype=np.uint64))
    assert len(index) == 0


def test_duplicate_of_existing_label_keeps_earlier_rows(rng):
    # Documented behaviour: a clash with a label already in the index is found
    # row by row, so the rows before the clash stay added.
    index = annex.FlatIndex(dim=DIM)
    data = rng.standard_normal((4, DIM)).astype(np.float32)
    index.add(data[0], labels=[7])
    with pytest.raises(ValueError, match="already present"):
        index.add(data[1:], labels=np.array([8, 7, 9], dtype=np.uint64))
    assert len(index) == 2
    labels, _ = index.search(data[1], k=1)
    assert labels[0, 0] == 8


def test_k_greater_than_len_raises(rng):
    index, data = make_index(rng, n=5)
    with pytest.raises(ValueError, match="k"):
        index.search(data[0], k=6)
    with pytest.raises(ValueError, match="k"):
        annex.FlatIndex(dim=DIM).search(data[0], k=1)


def test_negative_k_raises(rng):
    index, data = make_index(rng)
    with pytest.raises(ValueError, match="k"):
        index.search(data[0], k=-1)


# --- Agreement with NumPy brute force ----------------------------------------


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_matches_numpy_reference(rng, metric):
    n, nq, k = 500, 40, 10
    train = rng.standard_normal((n, DIM)).astype(np.float32)
    queries = rng.standard_normal((nq, DIM)).astype(np.float32)
    index = annex.FlatIndex(dim=DIM, metric=metric)
    index.add(train)

    labels, distances = index.search(queries, k=k)
    ref_labels, ref_distances = reference_search(train, queries, k, metric)

    np.testing.assert_array_equal(labels, ref_labels)
    # The core accumulates in float32 and the reference in float64.
    np.testing.assert_allclose(distances, ref_distances, rtol=1e-4, atol=1e-4)


# --- Concurrency -------------------------------------------------------------


def test_concurrent_search_matches_serial(rng):
    # search() releases the GIL, so these threads really run inside the core
    # at the same time. Each must still get exactly the serial answer.
    index, _ = make_index(rng, n=2000, dim=32)
    batches = [rng.standard_normal((50, 32)).astype(np.float32) for _ in range(16)]
    expected = [index.search(b, k=10) for b in batches]

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda b: index.search(b, k=10), batches))

    for (gl, gd), (el, ed) in zip(got, expected):
        np.testing.assert_array_equal(gl, el)
        np.testing.assert_array_equal(gd, ed)


def test_concurrent_add_and_search(rng):
    # add() takes the index exclusively, so interleaving it with searches must
    # neither crash nor lose vectors.
    dim = 8
    index = annex.FlatIndex(dim=dim)
    index.add(rng.standard_normal((10, dim)).astype(np.float32))
    chunks = [rng.standard_normal((100, dim)).astype(np.float32) for _ in range(20)]
    query = rng.standard_normal(dim).astype(np.float32)

    def add(chunk):
        index.add(chunk)

    def search(_):
        labels, _ = index.search(query, k=10)
        assert labels.shape == (1, 10)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(add, c) for c in chunks]
        futures += [pool.submit(search, i) for i in range(40)]
        for f in futures:
            f.result()

    assert len(index) == 10 + 20 * 100
    # Default labels were drawn under the lock, so all are distinct.
    labels, _ = index.search(query, k=len(index))
    assert len(np.unique(labels)) == len(index)
