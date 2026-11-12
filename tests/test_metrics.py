import numpy as np
import pytest

from engine_health_flow.metrics import pearson, point_metrics, rmse, smape


def test_rmse_per_column():
    y = np.array([[0.0, 1.0], [0.0, 1.0]])
    pred = np.array([[1.0, 1.0], [-1.0, 3.0]])
    assert rmse(y, pred) == pytest.approx([1.0, np.sqrt(2.0)])


def test_smape_equal_values_give_zero_even_at_zero():
    y = np.array([[0.0, 2.0], [0.0, -3.0]])
    assert smape(y, y) == pytest.approx([0.0, 0.0])


def test_smape_is_two_for_opposite_signs_and_symmetric():
    a = np.array([[1.0], [2.0]])
    assert smape(a, -a) == pytest.approx([2.0])
    b = np.array([[1.5], [3.0]])
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_smape_of_a_ten_percent_miss():
    # 2 * 0.1 / (1 + 1.1)
    assert smape(np.array([[1.0]]), np.array([[1.1]])) == pytest.approx([0.2 / 2.1])


def test_pearson_ignores_scale_and_offset():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 3))
    assert pearson(y, 3 * y + 1) == pytest.approx([1, 1, 1])
    assert pearson(y, -y) == pytest.approx([-1, -1, -1])


def test_pooled_numbers_are_computed_on_all_values_together():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(500, 4))
    pred = y + rng.normal(scale=[0.1, 0.1, 1.0, 1.0], size=y.shape)
    m = point_metrics(y, pred)
    assert m["pooled"]["rmse"] == pytest.approx(np.sqrt(((y - pred) ** 2).mean()))
    # pooling is not the mean of the columns
    assert m["pooled"]["rmse"] != pytest.approx(np.mean(m["rmse"]))
    assert len(m["rmse"]) == len(m["smape"]) == len(m["pearson"]) == 4


def test_point_metrics_is_json_friendly():
    import json

    y = np.random.default_rng(2).normal(size=(10, 2)).astype(np.float32)
    json.dumps(point_metrics(y, y * 0.9))
