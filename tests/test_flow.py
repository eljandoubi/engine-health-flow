import numpy as np
import pytest
import torch

from engine_health_flow.flow import FlowPosterior, fit_flow, flow_loss
from engine_health_flow.metrics import rmse
from engine_health_flow.scaling import Prepared
from engine_health_flow.training import fit
from engine_health_flow.windows import window_rows

SIGMA = 0.3


def toy_problem(n, generator):
    """Windows of 3 readings over 5 steps, and the mean of a target that depends on them."""
    window = torch.randn(n, 5, 3, generator=generator)
    mean = torch.stack([1.5 * window[:, :, 0].mean(1), -window[:, -1, 1]], dim=1)
    return window, mean


def test_flow_recovers_a_conditional_gaussian():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)

    def draw():
        window, mean = toy_problem(256, generator)
        return window, mean + SIGMA * torch.randn_like(mean)

    net = FlowPosterior(3, 2, hidden=32, field_hidden=64)
    fit(net, draw, flow_loss, steps=1200, lr=3e-3, log_every=10**9)

    window, mean = toy_problem(200, generator)
    draws = net.sample(window, n=256, steps=16)
    assert draws.shape == (200, 256, 2)
    assert (draws.mean(1) - mean).abs().mean() < 0.05
    assert draws.std(1).mean() == pytest.approx(SIGMA, rel=0.1)


def fitted_on_synthetic(synthetic):
    ds = synthetic(n_seq=8, length=60)
    train, test = ds.take(range(6)), ds.take([6, 7])
    model = fit_flow(
        Prepared.fit(train),
        train,
        window=10,
        steps=300,
        batch=64,
        repeats=2,
        hidden=32,
        field_hidden=64,
    )
    return model, train, test


def test_flow_model_tracks_synthetic_engines_better_than_the_mean(synthetic):
    model, train, test = fitted_on_synthetic(synthetic)
    rows = window_rows(test, window=10)
    err = rmse(test.health[rows], model.predict(test, rows)).mean()
    baseline = rmse(test.health[rows], np.broadcast_to(train.health.mean(0), (len(rows), 10)))
    assert err < 0.5 * baseline.mean()


def test_flow_samples_have_the_right_shape_and_follow_the_seed(synthetic):
    model, _, test = fitted_on_synthetic(synthetic)
    rows = window_rows(test, window=10)[:20]
    a = model.samples(test, rows, n=6, seed=3)
    assert a.shape == (20, 6, 10)
    assert np.array_equal(a, model.samples(test, rows, n=6, seed=3))
    assert not np.array_equal(a, model.samples(test, rows, n=6, seed=4))


def test_flow_refuses_rows_whose_window_leaves_the_sequence(synthetic):
    model, _, test = fitted_on_synthetic(synthetic)
    with pytest.raises(ValueError, match="window"):
        model.samples(test, np.array([test.starts[1] + 3]), n=2)
