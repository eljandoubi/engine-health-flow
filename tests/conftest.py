import numpy as np
import pandas as pd
import pytest

from engine_health_flow.data import HEALTH, SENSOR_COLUMNS, Dataset


def toy_frame(lengths=(5, 7, 4), shuffle=True) -> pd.DataFrame:
    """A csv-shaped frame where every cell tells which sequence, step and column it came from."""
    parts = []
    for i, n in enumerate(lengths):
        t = np.arange(n)
        part = pd.DataFrame({"sequence_id": f"seq-{i}", "timestep": t})
        for j, col in enumerate(HEALTH):
            part[col] = i * 100 + t + j / 10
        for k, col in enumerate(SENSOR_COLUMNS):
            part[col] = -(i * 100 + t + k / 10)
        part["maintenance"] = t == 2
        part["speed_change"] = "[True]"
        part["Cruise_DeckSMR__phase_type"] = "CR"
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    return frame.sample(frac=1, random_state=0) if shuffle else frame


@pytest.fixture
def toy_csv(tmp_path):
    path = tmp_path / "toy.csv"
    toy_frame().to_csv(path, index=False)
    return path


def make_synthetic(n_seq=8, length=120, noise=3e-4, seed=0) -> Dataset:
    """Health indicators that slowly worsen, and sensors that are a noisy, linear function of them.

    The last sensor is constant, like the LP_Nmech channels of the real data.
    """
    rng = np.random.default_rng(seed)
    mix = rng.normal(scale=0.5, size=(len(HEALTH), len(SENSOR_COLUMNS)))
    base = rng.uniform(100, 1000, size=len(SENSOR_COLUMNS))
    health, sensors = [], []
    for _ in range(n_seq):
        h = -np.cumsum(rng.uniform(0, 5e-4, size=(length, len(HEALTH))), axis=0)
        y = base * (1 + 3 * h @ mix + noise * rng.normal(size=(length, len(base))))
        y[:, -1] = 4031.0
        health.append(h)
        sensors.append(y)
    return Dataset(
        ids=np.array([f"s{i}" for i in range(n_seq)]),
        starts=np.arange(n_seq + 1) * length,
        health=np.concatenate(health).astype(np.float32),
        sensors=np.concatenate(sensors).astype(np.float32),
        maintenance=np.zeros(n_seq * length, dtype=bool),
    )


@pytest.fixture
def synthetic():
    return make_synthetic
