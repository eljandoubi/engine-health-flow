import numpy as np
import pandas as pd
import pytest

from engine_health_flow.data import HEALTH, SENSOR_COLUMNS


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
