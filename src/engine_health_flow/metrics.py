"""Point metrics per health indicator, as in the paper."""

import numpy as np


def rmse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(((y - pred) ** 2).mean(0))


def smape(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """2 |y - p| / (|y| + |p|) averaged over rows, so it lies in [0, 2].

    The paper does not say how it handles y = p = 0, here that counts as no error.
    """
    num = 2 * np.abs(y - pred)
    den = np.abs(y) + np.abs(pred)
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return ratio.mean(0)


def pearson(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    yc = y - y.mean(0)
    pc = pred - pred.mean(0)
    return (yc * pc).sum(0) / np.sqrt((yc**2).sum(0) * (pc**2).sum(0))


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Per indicator values, and the same three numbers on all indicators pooled together.

    The paper's "avg" column is computed on the pooled test values, not as the mean of the
    ten columns, so both are kept.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    flat_y, flat_p = y.reshape(-1, 1), pred.reshape(-1, 1)
    return {
        "rmse": rmse(y, pred).tolist(),
        "smape": smape(y, pred).tolist(),
        "pearson": pearson(y, pred).tolist(),
        "pooled": {
            "rmse": float(rmse(flat_y, flat_p)[0]),
            "smape": float(smape(flat_y, flat_p)[0]),
            "pearson": float(pearson(flat_y, flat_p)[0]),
        },
    }
