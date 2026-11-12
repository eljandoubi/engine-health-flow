"""Point metrics per health indicator as in the paper, and scores for sampled posteriors."""

from functools import partial

import numpy as np

SPREAD_GRID = np.geomspace(0.5, 2.0, 13)


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


def crps_samples(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score per indicator, from samples of the posterior.

    y is (rows, indicators) and samples (rows, n, indicators). This is the unbiased estimate: the
    mean distance to the truth minus half the mean distance between two different samples.
    """
    n = samples.shape[1]
    to_truth = np.abs(samples - y[:, None]).mean(1)
    ranks = (2 * np.arange(1, n + 1) - n - 1)[None, :, None]
    between = (ranks * np.sort(samples, axis=1)).sum(1) / (n * (n - 1))
    return (to_truth - between).mean(0)


def interval_scores(y: np.ndarray, samples: np.ndarray, level: float) -> np.ndarray:
    """Share of truths inside the central interval of the samples, and its mean width.

    Returns an array of shape (2, indicators).
    """
    low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
    return np.stack([((y >= low) & (y <= high)).mean(0), (high - low).mean(0)])


def rescale_spread(mean: np.ndarray, draws: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Move the draws towards or away from their mean, by one factor per indicator."""
    return mean[:, None] + scales * (draws - mean[:, None])


def spread_scales(
    y: np.ndarray, mean: np.ndarray, draws: np.ndarray, grid: np.ndarray = SPREAD_GRID
) -> np.ndarray:
    """The factor per indicator that gives the lowest CRPS when it scales the spread of the draws.

    A value above 1 means the draws were too tight, below 1 too wide.
    """
    scores = [crps_samples(y, rescale_spread(mean, draws, s)) for s in grid]
    return grid[np.argmin(scores, axis=0)]


def posterior_metrics(y: np.ndarray, samples: np.ndarray, levels=(0.5, 0.9), block=4096) -> dict:
    """CRPS, and coverage and width of central intervals, per indicator.

    Done on blocks of rows so that a few hundred thousand samples do not need all the memory.
    """
    y = np.asarray(y, dtype=np.float64)
    spans = [slice(i, i + block) for i in range(0, len(y), block)]
    sizes = [len(y[s]) for s in spans]

    def over_blocks(score):
        parts = [score(y[s], np.asarray(samples[s], dtype=np.float64)) for s in spans]
        return np.average(parts, axis=0, weights=sizes)

    crps = over_blocks(crps_samples)
    out = {"crps": crps.tolist(), "crps_mean": float(crps.mean()), "intervals": {}}
    for level in levels:
        covered, width = over_blocks(partial(interval_scores, level=level))
        out["intervals"][str(level)] = {"coverage": covered.tolist(), "width": width.tolist()}
    return out
