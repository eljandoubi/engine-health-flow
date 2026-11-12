"""An unscented Kalman filter on the learned surrogate, run on many sequences at once."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from engine_health_flow.data import Dataset
from engine_health_flow.scaling import Prepared
from engine_health_flow.surrogate import Surrogate, fit_surrogate
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows


@dataclass
class Dynamics:
    """How the standardised health indicators move from one row to the next."""

    drift: torch.Tensor  # mean step on an ordinary row
    cov: torch.Tensor  # covariance of that step
    jump: torch.Tensor  # mean step on a row where maintenance happened
    jump_cov: torch.Tensor
    start: torch.Tensor  # first row of a sequence
    start_cov: torch.Tensor


def _gaussian(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(values.mean(0)), torch.from_numpy(np.cov(values.T))


def estimate_dynamics(prep: Prepared, train: Dataset) -> Dynamics:
    z = prep.y(train).astype(np.float64)
    later = np.setdiff1d(np.arange(len(z)), train.starts[:-1])
    step = z[later] - z[later - 1]
    flagged = train.maintenance[later]
    drift, cov = _gaussian(step[~flagged])
    jump, jump_cov = _gaussian(step[flagged])
    start, start_cov = _gaussian(z[train.starts[:-1]])
    # the floor keeps it invertible when there are fewer sequences than indicators
    start_cov = start_cov + 1e-6 * torch.eye(len(start), dtype=start_cov.dtype)
    return Dynamics(drift, cov, jump, jump_cov, start, start_cov)


def _weights(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigma point weights for alpha = 1, kappa = 0 and beta = 2, the usual choice for Gaussians."""
    mean_w = torch.full((2 * n + 1,), 0.5 / n, dtype=torch.float64)
    mean_w[0] = 0.0
    cov_w = mean_w.clone()
    cov_w[0] = 2.0
    return mean_w, cov_w


def _correct(h, mean, cov, y, obs_cov, weights):
    """One measurement update of the batch: sigma points through h, then the Kalman gain."""
    batch, n = mean.shape
    mean_w, cov_w = weights
    eye = torch.eye(n, dtype=mean.dtype)
    root = torch.linalg.cholesky(n * cov + 1e-12 * eye)
    spread = root.transpose(-1, -2)  # row j is the j-th column of the root
    points = torch.cat([mean[:, None], mean[:, None] + spread, mean[:, None] - spread], dim=1)

    seen = h(points.reshape(-1, n)).reshape(batch, 2 * n + 1, -1)
    seen_mean = (mean_w[None, :, None] * seen).sum(1)
    dy = seen - seen_mean[:, None]
    dx = points - mean[:, None]
    innovation_cov = torch.einsum("k,bki,bkj->bij", cov_w, dy, dy) + obs_cov
    cross = torch.einsum("k,bki,bkj->bij", cov_w, dx, dy)
    gain = torch.linalg.solve(innovation_cov, cross.transpose(-1, -2)).transpose(-1, -2)

    new_mean = mean + torch.einsum("bij,bj->bi", gain, y - seen_mean)
    new_cov = cov - gain @ innovation_cov @ gain.transpose(-1, -2)
    return new_mean, 0.5 * (new_cov + new_cov.transpose(-1, -2))


@torch.no_grad()
def ukf(h, dyn, obs_cov, y, flagged, lengths, q_scale: float = 1.0):
    """Filtered mean and variance of the health indicators at every step of every sequence.

    h maps (points, indicators) to (points, sensors). y is (sequences, steps, sensors) and padded
    after the end of each sequence, lengths says where that is. flagged is (sequences, steps).
    """
    batch, steps, _ = y.shape
    weights = _weights(dyn.start.numel())
    mean = dyn.start.expand(batch, -1).clone()
    cov = dyn.start_cov.expand(batch, -1, -1).clone()
    means, variances = [], []
    for t in range(steps):
        if t > 0:
            jump = flagged[:, t, None]
            mean = mean + torch.where(jump, dyn.jump, dyn.drift)
            cov = cov + torch.where(jump[..., None], dyn.jump_cov, q_scale * dyn.cov)
        new_mean, new_cov = _correct(h, mean, cov, y[:, t], obs_cov, weights)
        live = (t < lengths)[:, None]
        mean = torch.where(live, new_mean, mean)
        cov = torch.where(live[..., None], new_cov, cov)
        means.append(mean)
        variances.append(torch.diagonal(cov, dim1=-2, dim2=-1))
    return torch.stack(means, dim=1), torch.stack(variances, dim=1)


def _pad(x: np.ndarray, ds: Dataset) -> torch.Tensor:
    out = np.zeros((ds.n_seq, ds.lengths.max(), *x.shape[1:]), dtype=x.dtype)
    for i, n in enumerate(ds.lengths):
        out[i, :n] = x[ds.rows(i)]
    return torch.from_numpy(out)


def _unpad(a: torch.Tensor, ds: Dataset) -> np.ndarray:
    return np.concatenate([a[i, :n].numpy() for i, n in enumerate(ds.lengths)])


class UKFModel:
    """Filters whole sequences, so predict() reads the estimate at the rows it is given."""

    def __init__(
        self,
        prep: Prepared,
        surrogate: Surrogate,
        dyn: Dynamics,
        obs_cov: torch.Tensor,
        q_scale: float = 1.0,
    ):
        self.prep, self.surrogate, self.dyn = prep, surrogate, dyn
        self.obs_cov, self.q_scale = obs_cov, q_scale

    def filtered(
        self, ds: Dataset, keep: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the health indicators on every row, in raw units.

        keep lists the sensor channels that can be read, the others are left out of the update.
        """
        y = _pad(self.prep.x(ds).astype(np.float64), ds)
        h, obs_cov = self.surrogate, self.obs_cov
        if keep is not None:
            keep = torch.from_numpy(keep)
            y, obs_cov = y[..., keep], obs_cov[keep][:, keep]

            def h(state):
                return self.surrogate(state)[:, keep]

        flagged = _pad(ds.maintenance, ds)
        lengths = torch.from_numpy(ds.lengths)
        mean, var = ukf(h, self.dyn, obs_cov, y, flagged, lengths, self.q_scale)
        scale = self.prep.health.scale
        return self.prep.health.inverse(_unpad(mean, ds)), np.sqrt(_unpad(var, ds)) * scale

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return self.filtered(ds)[0][rows]

    def posterior(
        self, ds: Dataset, rows: np.ndarray, n: int = 64, seed: int = 0, keep=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The filtered mean at the rows and n Gaussian draws around it, (rows, n, indicators)."""
        mean, std = self.filtered(ds, keep)
        noise = np.random.default_rng(seed).standard_normal((len(rows), n, mean.shape[1]))
        return mean[rows], mean[rows, None] + std[rows, None] * noise


def fit_ukf(
    prep: Prepared,
    train: Dataset,
    val: Dataset,
    seed: int = 0,
    surrogate: Surrogate | None = None,
    q_scales: tuple[float, ...] = (16.0, 64.0, 256.0),
    r_scales: tuple[float, ...] = (1.5, 2.0, 3.0),
    **surrogate_options,
) -> UKFModel:
    """Surrogate and step statistics come from train, the two noise scales from val.

    The best scales are well above 1: the filter needs room for the surrogate's small errors and
    for health indicators that do not move like a plain random walk.
    """
    if surrogate is None:
        surrogate = fit_surrogate(prep, train, seed=seed, **surrogate_options)
    dyn = estimate_dynamics(prep, train)
    base = torch.diag(torch.from_numpy(surrogate.noise.astype(np.float64)) ** 2)
    rows = window_rows(val, WINDOW, EVAL_STRIDE)

    best_error, best = np.inf, None
    for q_scale, r_scale in product(q_scales, r_scales):
        model = UKFModel(prep, surrogate, dyn, base * r_scale**2, q_scale)
        error = np.sqrt(((val.health[rows] - model.predict(val, rows)) ** 2).mean())
        if error < best_error:
            best_error, best = error, model
    return best
