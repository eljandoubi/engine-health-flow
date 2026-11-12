import numpy as np

from engine_health_flow.hybrid import FilteredInputs, fit_hybrid
from engine_health_flow.metrics import rmse
from engine_health_flow.scaling import Prepared
from engine_health_flow.windows import window_rows


def test_filtered_inputs_append_the_filter_to_the_sensors(fitted_ukf):
    model, _, test = fitted_ukf
    inputs = FilteredInputs(model.prep, model)
    x = inputs.x(test)
    mean, std = model.filtered(test)
    n = len(model.prep.cols)
    np.testing.assert_allclose(x[:, :n], model.prep.x(test))
    np.testing.assert_allclose(x[:, n : n + 10], model.prep.health.transform(mean), atol=1e-5)
    np.testing.assert_allclose(x[:, n + 10 :], np.log(std / model.prep.health.scale), atol=1e-5)

    rows = np.array([3, 70, 150])
    np.testing.assert_array_equal(inputs.x(test, rows), x[rows])
    assert inputs.y(test).shape == test.health.shape


def test_hybrid_flow_reads_the_filter_and_tracks_synthetic_engines(synthetic):
    ds = synthetic(n_seq=10, length=120, jump_every=40)
    train, val, test = ds.take(range(6)), ds.take([6, 7]), ds.take([8, 9])
    prep = Prepared.fit(train)
    small_ukf = {
        "q_scales": (1.0,),
        "r_scales": (1.0,),
        "steps": 300,
        "batch": 256,
        "hidden": 32,
        "depth": 2,
    }
    model = fit_hybrid(
        prep,
        train,
        val,
        ukf_options=small_ukf,
        window=10,
        steps=300,
        batch=64,
        repeats=2,
        hidden=32,
        field_hidden=64,
    )
    assert model.net.encoder.gru.input_size == len(prep.cols) + 2 * 10

    rows = window_rows(test, window=10)
    err = rmse(test.health[rows], model.predict(test, rows)).mean()
    baseline = rmse(test.health[rows], np.broadcast_to(train.health.mean(0), (len(rows), 10)))
    assert err < 0.5 * baseline.mean()
