"""A flow matching posterior that also reads the output of an unscented Kalman filter."""

import numpy as np

from engine_health_flow.data import Dataset
from engine_health_flow.flow import FlowModel, fit_flow
from engine_health_flow.scaling import Prepared
from engine_health_flow.ukf import UKFModel, fit_ukf


class FilteredInputs:
    """Standardised sensors with the filter's health estimate and its log std on every row.

    The filter has seen the whole sequence up to that row, far more than a window of 50 steps, so
    the flow gets a long memory for free. This stands in for Prepared, which is all the flow uses.
    """

    def __init__(self, prep: Prepared, ukf: UKFModel):
        self.prep, self.ukf, self.health = prep, ukf, prep.health

    def x(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        mean, std = self.ukf.filtered(ds)
        filtered = [self.health.transform(mean), np.log(std / self.health.scale)]
        x = np.concatenate([self.prep.x(ds), *filtered], axis=1).astype(np.float32)
        return x if rows is None else x[rows]

    def y(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        return self.prep.y(ds, rows)


def fit_hybrid(
    prep: Prepared,
    train: Dataset,
    val: Dataset,
    seed: int = 0,
    ukf_options: dict | None = None,
    **flow_options,
) -> FlowModel:
    ukf = fit_ukf(prep, train, val, seed=seed, **(ukf_options or {}))
    return fit_flow(FilteredInputs(prep, ukf), train, seed=seed, **flow_options)
