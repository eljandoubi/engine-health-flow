import numpy as np
import pytest
from scipy.stats import norm

from engine_health_flow.metrics import (
    crps_samples,
    interval_scores,
    pearson,
    point_metrics,
    posterior_metrics,
    rescale_spread,
    rmse,
    smape,
    spread_scales,
)


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


def gaussian_crps(y, mean, std):
    z = (y - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_crps_of_small_hand_made_samples():
    one = np.array([[1.0]])
    # three evenly spread samples centred on the truth
    assert crps_samples(one, np.array([[[0.0], [1.0], [2.0]]])) == pytest.approx([0.0])
    # no spread at all: the score is the distance to the truth
    assert crps_samples(one, np.array([[[3.0], [3.0], [3.0]]])) == pytest.approx([2.0])


def test_crps_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(0)
    mean, std = rng.normal(size=(400, 2)), rng.uniform(0.5, 2.0, size=(400, 2))
    y = mean + std * rng.normal(size=mean.shape)
    samples = mean[:, None] + std[:, None] * rng.normal(size=(400, 500, 2))
    want = gaussian_crps(y, mean, std).mean(0)
    assert crps_samples(y, samples) == pytest.approx(want, rel=0.02)


def test_crps_prefers_the_calibrated_spread():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(600, 1))

    def draw(scale):
        return scale * rng.normal(size=(600, 200, 1))

    best = crps_samples(y, draw(1.0))
    assert best < crps_samples(y, draw(0.2))
    assert best < crps_samples(y, draw(4.0))


def test_interval_scores_on_a_uniform_grid():
    grid = np.linspace(0, 1, 101)[None, :, None] * np.ones((3, 1, 1))
    y = np.array([[0.5], [0.9], [1.5]])
    cover50, width50 = interval_scores(y, grid, 0.5)
    cover90, width90 = interval_scores(y, grid, 0.9)
    # the first truth is inside both, the second only inside the 90 % interval, the third in none
    assert cover50 == pytest.approx([1 / 3])
    assert cover90 == pytest.approx([2 / 3])
    assert width50 == pytest.approx([0.5])
    assert width90 == pytest.approx([0.9])


def test_posterior_metrics_in_blocks_equals_in_one_go():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(50, 3))
    samples = y[:, None] + rng.normal(size=(50, 40, 3))
    whole = posterior_metrics(y, samples, block=50)
    blocks = posterior_metrics(y, samples, block=7)
    assert blocks["crps"] == pytest.approx(whole["crps"])
    assert blocks["intervals"]["0.9"]["coverage"] == pytest.approx(
        whole["intervals"]["0.9"]["coverage"]
    )


def test_rescaling_keeps_the_mean_and_scales_the_spread():
    rng = np.random.default_rng(5)
    draws = rng.normal(size=(20, 300, 2))
    mean = draws.mean(1)
    out = rescale_spread(mean, draws, np.array([2.0, 0.5]))
    np.testing.assert_allclose(out.mean(1), mean)
    np.testing.assert_allclose(out.std(1), draws.std(1) * [2.0, 0.5])


def test_spread_scales_find_draws_that_are_too_tight_or_too_wide():
    rng = np.random.default_rng(6)
    y = rng.normal(size=(2000, 3))
    mean = np.zeros_like(y)
    # truths have spread 1, the draws have 0.5, 2 and 1
    draws = rng.normal(size=(2000, 100, 3)) * [0.5, 2.0, 1.0]
    scales = spread_scales(y, mean, draws)
    assert scales == pytest.approx([2.0, 0.5, 1.0], rel=0.15)


def test_posterior_metrics_is_json_friendly():
    import json

    y = np.zeros((5, 2))
    json.dumps(posterior_metrics(y, np.random.default_rng(4).normal(size=(5, 8, 2))))
