"""Standardising sensors and health indicators with statistics of the training sequences."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from engine_health_flow.data import Dataset


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Self:
        mean = x.mean(0, dtype=np.float64)
        scale = x.std(0, dtype=np.float64)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale + self.mean


def informative_columns(x: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Indices of the columns that vary. In the data three LP_Nmech channels do not."""
    mean = np.abs(x.mean(0, dtype=np.float64))
    std = x.std(0, dtype=np.float64)
    return np.flatnonzero(std > rel_tol * np.maximum(mean, 1e-12))


@dataclass
class Prepared:
    """Everything the models need to turn a dataset into standardised arrays."""

    cols: np.ndarray  # sensor columns that are kept
    sensors: Scaler
    health: Scaler

    @classmethod
    def fit(cls, train: Dataset) -> Self:
        cols = informative_columns(train.sensors)
        return cls(cols, Scaler.fit(train.sensors[:, cols]), Scaler.fit(train.health))

    def x(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        sensors = ds.sensors if rows is None else ds.sensors[rows]
        return self.sensors.transform(sensors[:, self.cols])

    def y(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        health = ds.health if rows is None else ds.health[rows]
        return self.health.transform(health)
