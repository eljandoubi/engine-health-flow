import numpy as np
import torch

from engine_health_flow.metrics import rmse
from engine_health_flow.scaling import Prepared
from engine_health_flow.ukf import Dynamics, estimate_dynamics, fit_ukf, ukf
from engine_health_flow.windows import window_rows


def random_dynamics(n: int, seed: int = 0) -> Dynamics:
    rng = np.random.default_rng(seed)

    def cov(scale):
        a = rng.normal(size=(n, n))
        return torch.from_numpy(scale * (a @ a.T / n + np.eye(n)))

    def vec(scale):
        return torch.from_numpy(scale * rng.normal(size=n))

    return Dynamics(vec(0.01), cov(0.01), vec(1.0), cov(0.5), vec(0.1), cov(0.2))


def kalman(a, dyn, obs_cov, y, flagged):
    """The textbook filter for y = a x + noise, to compare the unscented one with."""
    mean, cov = dyn.start.numpy(), dyn.start_cov.numpy()
    means, variances = [], []
    for t in range(len(y)):
        if t > 0:
            mean = mean + (dyn.jump if flagged[t] else dyn.drift).numpy()
            cov = cov + (dyn.jump_cov if flagged[t] else dyn.cov).numpy()
        s = a @ cov @ a.T + obs_cov
        gain = cov @ a.T @ np.linalg.inv(s)
        mean = mean + gain @ (y[t] - a @ mean)
        cov = cov - gain @ s @ gain.T
        means.append(mean)
        variances.append(np.diag(cov))
    return np.array(means), np.array(variances)


def test_ukf_equals_the_kalman_filter_when_the_sensors_are_linear():
    n, m, steps = 4, 6, 30
    rng = np.random.default_rng(1)
    a = rng.normal(size=(m, n))
    obs_cov = np.diag(rng.uniform(0.01, 0.1, size=m))
    y = rng.normal(size=(2, steps, m))
    flagged = np.zeros((2, steps), dtype=bool)
    flagged[0, [7, 20]] = True
    flagged[1, 12] = True
    dyn = random_dynamics(n)
    lengths = [steps, 18]  # the second sequence ends early and is padded

    mean, var = ukf(
        lambda z: z @ torch.from_numpy(a).T,
        dyn,
        torch.from_numpy(obs_cov),
        torch.from_numpy(y),
        torch.from_numpy(flagged),
        torch.tensor(lengths),
    )
    for i, length in enumerate(lengths):
        want_mean, want_var = kalman(a, dyn, obs_cov, y[i, :length], flagged[i])
        np.testing.assert_allclose(mean[i, :length].numpy(), want_mean, atol=1e-8)
        np.testing.assert_allclose(var[i, :length].numpy(), want_var, atol=1e-8)


def test_a_flagged_row_lets_the_estimate_jump():
    n = 2
    zero, eye = torch.zeros(n, dtype=torch.float64), torch.eye(n, dtype=torch.float64)
    dyn = Dynamics(zero, 1e-6 * eye, zero + 2.0, 0.1 * eye, zero, 1e-6 * eye)
    y = torch.zeros(1, 20, n, dtype=torch.float64)
    y[0, 10:] = 5.0
    plain = torch.zeros(1, 20, dtype=torch.bool)
    flagged = plain.clone()
    flagged[0, 10] = True
    args = (lambda z: z, dyn, 1e-4 * eye, y)

    with_flag, _ = ukf(*args, flagged, torch.tensor([20]))
    without, _ = ukf(*args, plain, torch.tensor([20]))
    assert (with_flag[0, 10] - 5.0).abs().max() < 0.05
    assert without[0, 10].max() < 1.0  # a random walk with tiny steps barely moves in one row


def test_dynamics_come_from_steps_inside_sequences(synthetic):
    ds = synthetic(n_seq=6, length=100, jump_every=30)
    prep = Prepared.fit(ds)
    z = prep.y(ds).astype(np.float64)
    ordinary, jumps = [], []
    for i in range(ds.n_seq):
        for row in range(ds.starts[i] + 1, ds.starts[i + 1]):
            (jumps if ds.maintenance[row] else ordinary).append(z[row] - z[row - 1])

    dyn = estimate_dynamics(prep, ds)
    np.testing.assert_allclose(dyn.drift.numpy(), np.mean(ordinary, axis=0))
    np.testing.assert_allclose(dyn.cov.numpy(), np.cov(np.array(ordinary).T))
    np.testing.assert_allclose(dyn.jump.numpy(), np.mean(jumps, axis=0))
    np.testing.assert_allclose(dyn.start.numpy(), z[ds.starts[:-1]].mean(0))
    assert dyn.jump.min() > 5 * dyn.drift.abs().max()


def fitted_ukf(synthetic):
    ds = synthetic(n_seq=10, length=120, jump_every=40)
    train, val, test = ds.take(range(6)), ds.take([6, 7]), ds.take([8, 9])
    model = fit_ukf(
        Prepared.fit(train),
        train,
        val,
        q_scales=(1.0,),
        r_scales=(1.0,),
        steps=300,
        batch=256,
        hidden=32,
        depth=2,
    )
    return model, train, test


def test_ukf_tracks_synthetic_engines_better_than_the_mean(synthetic):
    model, train, test = fitted_ukf(synthetic)
    rows = window_rows(test)
    mean, std = model.filtered(test)
    assert mean.shape == std.shape == test.health.shape
    assert (std > 0).all()

    err = rmse(test.health[rows], model.predict(test, rows)).mean()
    baseline = rmse(test.health[rows], np.broadcast_to(train.health.mean(0), (len(rows), 10)))
    assert err < 0.5 * baseline.mean()


def test_ukf_posterior_draws_follow_the_filtered_gaussian(synthetic):
    model, _, test = fitted_ukf(synthetic)
    rows = window_rows(test)[:30]
    mean, draws = model.posterior(test, rows, n=4000)
    want_mean, want_std = model.filtered(test)
    assert draws.shape == (30, 4000, 10)
    np.testing.assert_allclose(mean, want_mean[rows])
    assert np.all(np.abs(draws.mean(1) - mean) < 5 * want_std[rows] / np.sqrt(4000))
    np.testing.assert_allclose(draws.std(1), want_std[rows], rtol=0.1)
