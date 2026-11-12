"""Models that read one row of sensors at a time: ridge regression and gradient boosting."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from engine_health_flow.data import Dataset
from engine_health_flow.scaling import Prepared


class RowModel:
    """An sklearn regressor trained on standardised sensors and health indicators."""

    def __init__(self, prep: Prepared, regressor):
        self.prep = prep
        self.regressor = regressor

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        z = self.regressor.predict(self.prep.x(ds, rows))
        return self.prep.health.inverse(z)


def fit_ridge(prep: Prepared, train: Dataset, alpha: float = 1.0) -> RowModel:
    return RowModel(prep, Ridge(alpha=alpha).fit(prep.x(train), prep.y(train)))


def fit_boosting(
    prep: Prepared, train: Dataset, max_iter: int = 200, n_jobs: int = -1, seed: int = 0
) -> RowModel:
    one = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
    forest = MultiOutputRegressor(one, n_jobs=n_jobs)
    return RowModel(prep, forest.fit(prep.x(train), prep.y(train)))
