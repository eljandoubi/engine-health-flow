import numpy as np

from engine_health_flow.baselines import fit_boosting, fit_ridge
from engine_health_flow.metrics import rmse
from engine_health_flow.scaling import Prepared


def split(ds):
    return ds.take(range(0, 6)), ds.take([6, 7])


def test_ridge_recovers_a_nearly_linear_problem(synthetic):
    train, test = split(synthetic())
    model = fit_ridge(Prepared.fit(train), train)
    rows = np.arange(len(test.health))
    err = rmse(test.health, model.predict(test, rows))
    assert np.all(err < 0.3 * test.health.std(0))


def test_boosting_predicts_in_the_original_units(synthetic):
    train, test = split(synthetic())
    model = fit_boosting(Prepared.fit(train), train, max_iter=30, n_jobs=1)
    rows = np.arange(0, len(test.health), 3)
    pred = model.predict(test, rows)
    assert pred.shape == (len(rows), test.health.shape[1])
    # same order of magnitude as the targets, so the inverse scaling was applied
    assert np.abs(pred).max() < 10 * np.abs(test.health).max()
    assert rmse(test.health[rows], pred).mean() < test.health.std()


def test_prediction_only_uses_the_requested_rows(synthetic):
    train, test = split(synthetic())
    model = fit_ridge(Prepared.fit(train), train)
    rows = np.array([10, 50, 200])
    assert np.allclose(
        model.predict(test, rows), model.predict(test, np.arange(240))[rows], atol=1e-6
    )
