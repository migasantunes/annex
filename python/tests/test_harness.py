"""Tests for the benchmark harness in benchmarks/ (metrics, datasets, engines, CLI)."""

import json

import numpy as np
import pytest
from conftest import reference_search

from benchmarks import datasets, engines, metrics, run

# --- recall_at_k on hand-made examples ---------------------------------------

GT = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)


def test_recall_perfect():
    assert metrics.recall_at_k(GT.copy(), GT, k=3) == 1.0


def test_recall_ignores_order_within_top_k():
    found = np.array([[3, 1, 2], [6, 5, 4]], dtype=np.uint64)
    assert metrics.recall_at_k(found, GT, k=3) == 1.0


def test_recall_zero():
    found = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.uint64)
    assert metrics.recall_at_k(found, GT, k=3) == 0.0


def test_recall_partial():
    # Row 0 finds 2 of 3, row 1 finds 1 of 3: mean (2/3 + 1/3) / 2 = 0.5.
    found = np.array([[1, 2, 9], [6, 10, 11]], dtype=np.uint64)
    assert metrics.recall_at_k(found, GT, k=3) == pytest.approx(0.5)


def test_recall_uses_only_first_k_columns():
    # Only the first column counts at k=1: row 0 matches, row 1 does not.
    found = np.array([[1, 0, 0], [5, 4, 6]], dtype=np.uint64)
    assert metrics.recall_at_k(found, GT, k=1) == pytest.approx(0.5)


def test_recall_rejects_bad_k():
    with pytest.raises(ValueError):
        metrics.recall_at_k(GT, GT, k=4)
    with pytest.raises(ValueError):
        metrics.recall_at_k(GT, GT, k=0)


def test_latency_stats():
    stats = metrics.latency_stats(np.array([0.001, 0.002, 0.003, 0.004]))
    assert stats.qps == pytest.approx(4 / 0.010)
    assert stats.mean_ms == pytest.approx(2.5)
    assert stats.p50_ms == pytest.approx(2.5)


def test_timer_freezes_after_block():
    with metrics.timer() as elapsed:
        pass
    first = elapsed()
    assert first >= 0.0 and elapsed() == first


# --- Datasets ----------------------------------------------------------------


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_synthetic_ground_truth_is_exact(metric):
    data = datasets.load("synthetic", n=300, dim=8, queries=20, k=5, metric=metric, seed=1)
    assert data.train.shape == (300, 8) and data.train.dtype == np.float32
    assert data.queries.shape == (20, 8) and data.queries.dtype == np.float32
    assert data.ground_truth.shape == (20, 5) and data.ground_truth.dtype == np.uint64
    expected, _ = reference_search(data.train, data.queries, 5, metric)
    np.testing.assert_array_equal(data.ground_truth, expected)


def test_ground_truth_independent_of_chunk_size(rng):
    train = rng.standard_normal((100, 4)).astype(np.float32)
    queries = rng.standard_normal((30, 4)).astype(np.float32)
    a = datasets.exact_ground_truth(train, queries, 7, "l2", chunk=7)
    b = datasets.exact_ground_truth(train, queries, 7, "l2", chunk=1000)
    np.testing.assert_array_equal(a, b)


def test_synthetic_is_seeded():
    kwargs = dict(n=50, dim=4, queries=5, k=3, metric="l2")
    a = datasets.load("synthetic", seed=3, **kwargs)
    b = datasets.load("synthetic", seed=3, **kwargs)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.ground_truth, b.ground_truth)


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.load("nope")


# --- Engines -----------------------------------------------------------------


@pytest.mark.parametrize("engine_name", sorted(engines.ENGINES))
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_exact_engines_match_ground_truth(engine_name, metric):
    data = datasets.load("synthetic", n=400, dim=16, queries=25, k=10, metric=metric, seed=2)
    engine = engines.create(engine_name)
    engine.build(data.train, data.metric)
    found = engine.search(data.queries, 10)
    assert found.shape == (25, 10) and found.dtype == np.uint64
    assert metrics.recall_at_k(found, data.ground_truth, 10) == 1.0


def test_exact_engines_have_no_query_params():
    with pytest.raises(ValueError, match="ef_search"):
        engines.create("annex-flat").set_query_param("ef_search", 64)


def test_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        engines.create("nope")


# --- CLI end to end ----------------------------------------------------------


def test_run_writes_result(tmp_path, capsys):
    out = tmp_path / "result.json"
    argv = ["--engine", "annex-flat", "--dataset", "synthetic", "--n", "300", "--dim", "8"]
    argv += ["--queries", "10", "--k", "5", "--output", str(out)]
    assert run.main(argv) == 0

    result = json.loads(out.read_text())
    assert result["engine"] == "annex-flat" and result["dataset"] == "synthetic"
    assert result["parameters"] == {
        "n": 300, "dim": 8, "queries": 10, "k": 5, "metric": "l2", "seed": 0
    }
    assert result["recall"] == 1.0
    assert result["qps"] > 0 and result["build_seconds"] >= 0
    assert set(result["latency_ms"]) == {"mean", "p50", "p95", "p99"}
    env = result["environment"]
    for key in ("platform", "cpu_model", "python", "numpy", "annex", "annex_kernel",
                "git_commit", "timestamp_utc"):
        assert key in env
    assert "recall@5 1.0000" in capsys.readouterr().out


def test_run_rejects_k_above_n():
    with pytest.raises(SystemExit):
        run.parse_args(["--engine", "annex-flat", "--dataset", "synthetic", "--n", "5", "--k", "6"])
