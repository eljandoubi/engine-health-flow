"""Losing a whole flight phase of sensors: the ukf skips them, the flow trains with dropout."""

import numpy as np
import torch

from engine_health_flow.data import PHASES, SENSORS, Dataset
from engine_health_flow.experiment import environment
from engine_health_flow.flow import FlowModel, fit_flow
from engine_health_flow.metrics import posterior_metrics
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.ukf import fit_ukf
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows


def phase_of_channel(prep: Prepared) -> np.ndarray:
    """Flight phase, as an index into PHASES, of every sensor channel the models read."""
    return prep.cols // len(SENSORS)


class PhaseFlags:
    """Standardised sensors with one flag per flight phase appended: 1 if the phase is there.

    A missing phase has its channels set to 0, the training mean, and its flag to 0.
    This stands in for Prepared, which is all the flow uses.
    """

    def __init__(self, prep: Prepared, missing: int | None = None):
        self.prep, self.missing, self.health = prep, missing, prep.health

    def x(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        x = self.prep.x(ds)
        flags = np.ones((len(x), len(PHASES)), dtype=x.dtype)
        if self.missing is not None:
            x[:, phase_of_channel(self.prep) == self.missing] = 0.0
            flags[:, self.missing] = 0.0
        x = np.concatenate([x, flags], axis=1)
        return x if rows is None else x[rows]

    def y(self, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        return self.prep.y(ds, rows)


def drop_phases(prep: Prepared, p: float = 0.25):
    """Augmentation for windows of PhaseFlags inputs: each window loses each phase with chance p.

    A phase is lost for the whole window, as it would be when its sensors fail.
    """
    phase = torch.from_numpy(phase_of_channel(prep))
    n_channels = len(phase)

    def augment(window):
        present = torch.rand(len(window), len(PHASES)) >= p
        window = window.clone()
        window[:, :, :n_channels] *= present[:, phase][:, None, :]
        window[:, :, n_channels:] = present[:, None, :].to(window.dtype)
        return window

    return augment


def _scores(truth: np.ndarray, mean: np.ndarray, draws: np.ndarray) -> dict:
    posterior = posterior_metrics(truth, draws)
    return {
        "rmse": float(np.sqrt(((truth - mean) ** 2).mean())),
        "crps_mean": posterior["crps_mean"],
        "coverage_90": float(np.mean(posterior["intervals"]["0.9"]["coverage"])),
    }


def run_missing(
    ds: Dataset, fold: int = 0, seed: int = 0, ukf_options=None, flow_options=None
) -> dict:
    """Test scores of the ukf and of a flow trained with phase dropout, for each lost phase.

    The ukf and the flow are fitted once. Then every phase is taken away in turn at test time
    (and none, for reference). The ukf keeps the noise scales it chose with all sensors.
    """
    split = make_folds(ds.n_seq)[fold]
    train, val, test = ds.take(split.train), ds.take(split.val), ds.take(split.test)
    prep = Prepared.fit(train)
    filter_ = fit_ukf(prep, train, val, seed=seed, **(ukf_options or {}))
    flow = fit_flow(
        PhaseFlags(prep), train, seed=seed, augment=drop_phases(prep), **(flow_options or {})
    )

    rows = window_rows(test, WINDOW, EVAL_STRIDE)
    truth = test.health[rows]
    phase = phase_of_channel(prep)
    conditions = {}
    for missing in [None, *range(len(PHASES))]:
        label = "none" if missing is None else PHASES[missing]
        keep = None if missing is None else np.flatnonzero(phase != missing)
        reader = FlowModel(PhaseFlags(prep, missing), flow.net, flow.device, WINDOW)
        conditions[label] = {
            "ukf": _scores(truth, *filter_.posterior(test, rows, keep=keep)),
            "flow": _scores(truth, *reader.posterior(test, rows)),
        }
        print(f"{label}: done", flush=True)
    return {
        "fold": fold,
        "seed": seed,
        "scored_rows": len(rows),
        "environment": environment(),
        "conditions": conditions,
    }
