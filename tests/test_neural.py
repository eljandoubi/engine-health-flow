import numpy as np
import pytest

from engine_health_flow.metrics import rmse
from engine_health_flow.neural import MLP, fit_gru, fit_mlp
from engine_health_flow.scaling import Prepared
from engine_health_flow.windows import window_rows


def test_mlp_output_shape():
    import torch

    out = MLP(25, 10, hidden=16, depth=2)(torch.zeros(7, 25))
    assert out.shape == (7, 10)


def test_mlp_beats_the_mean_on_a_synthetic_problem(synthetic):
    ds = synthetic()
    train, test = ds.take(range(6)), ds.take([6, 7])
    model = fit_mlp(Prepared.fit(train), train, steps=300, batch=256, hidden=32, depth=2)
    rows = np.arange(len(test.health))
    pred = model.predict(test, rows)
    assert pred.shape == test.health.shape
    baseline = np.broadcast_to(train.health.mean(0), test.health.shape)
    assert rmse(test.health, pred).mean() < 0.5 * rmse(test.health, baseline).mean()


def test_mlp_is_reproducible_for_a_seed(synthetic):
    ds = synthetic(n_seq=3)
    prep = Prepared.fit(ds)
    a = fit_mlp(prep, ds, steps=20, batch=64, hidden=8, depth=1, seed=5)
    b = fit_mlp(prep, ds, steps=20, batch=64, hidden=8, depth=1, seed=5)
    rows = np.arange(0, len(ds.health), 7)
    assert np.allclose(a.predict(ds, rows), b.predict(ds, rows))


def test_gru_beats_the_mean_on_windows(synthetic):
    ds = synthetic()
    train, test = ds.take(range(6)), ds.take([6, 7])
    model = fit_gru(Prepared.fit(train), train, window=10, steps=200, batch=128, hidden=16)
    rows = window_rows(test, window=10)
    pred = model.predict(test, rows)
    assert pred.shape == (len(rows), test.health.shape[1])
    baseline = np.broadcast_to(train.health.mean(0), pred.shape)
    assert rmse(test.health[rows], pred).mean() < 0.7 * rmse(test.health[rows], baseline).mean()


def test_gru_refuses_rows_whose_window_leaves_the_sequence(synthetic):
    ds = synthetic(n_seq=3)
    model = fit_gru(Prepared.fit(ds), ds, window=10, steps=5, batch=32, hidden=8)
    with pytest.raises(ValueError, match="window"):
        model.predict(ds, np.array([ds.starts[1] + 3]))  # 4th step of the second sequence
