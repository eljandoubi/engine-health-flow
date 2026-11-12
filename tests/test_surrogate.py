import numpy as np
import torch

from engine_health_flow.data import Dataset
from engine_health_flow.scaling import Prepared
from engine_health_flow.surrogate import fit_surrogate

NOISE = 0.1  # of the spread of each sensor


def linear_world(n=3000, n_sensors=12, seed=0) -> Dataset:
    """Independent health indicators and sensors that are a linear mix of them plus noise."""
    rng = np.random.default_rng(seed)
    health = rng.normal(scale=0.01, size=(n, 10))
    signal = health @ rng.normal(size=(10, n_sensors))
    sensors = signal + NOISE * signal.std(0) * rng.normal(size=signal.shape)
    return Dataset(
        ids=np.array(["a"]),
        starts=np.array([0, n]),
        health=health.astype(np.float32),
        sensors=(500 + sensors).astype(np.float32),
        maintenance=np.zeros(n, dtype=bool),
    )


def test_surrogate_noise_estimate_matches_the_noise_that_was_put_in():
    ds = linear_world()
    prep = Prepared.fit(ds)
    sur = fit_surrogate(prep, ds, steps=600, batch=256, hidden=32, depth=2)
    pred = sur(torch.from_numpy(prep.y(ds)).double()).detach().numpy()
    assert pred.shape == (3000, 12)
    # the net cannot do better than the noise, and with this much data it gets close to it
    assert abs(sur.noise.mean() - NOISE) < 0.04
    assert np.abs(pred - prep.x(ds)).mean() < 0.15


def test_surrogate_runs_in_double_precision(synthetic):
    ds = synthetic(n_seq=3)
    sur = fit_surrogate(Prepared.fit(ds), ds, steps=5, batch=32, hidden=8, depth=1)
    out = sur(torch.zeros(4, 10, dtype=torch.float64))
    assert out.dtype == torch.float64
