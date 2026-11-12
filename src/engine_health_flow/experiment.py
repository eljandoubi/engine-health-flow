"""Fit the baselines on one fold and collect their metrics."""

import platform
import subprocess
import time

import numpy as np
import sklearn
import torch

from engine_health_flow.baselines import fit_boosting, fit_ridge
from engine_health_flow.data import Dataset
from engine_health_flow.metrics import point_metrics
from engine_health_flow.neural import fit_gru, fit_mlp
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.windows import WINDOW, window_rows

MODELS = ("ridge", "boosting", "mlp", "gru")
EVAL_STRIDE = 5


def fit_model(name: str, prep: Prepared, train: Dataset, seed: int, **options):
    if name == "ridge":
        return fit_ridge(prep, train, **options)
    if name == "boosting":
        return fit_boosting(prep, train, seed=seed, **options)
    if name == "mlp":
        return fit_mlp(prep, train, seed=seed, **options)
    if name == "gru":
        return fit_gru(prep, train, seed=seed, **options)
    raise ValueError(f"unknown model {name!r}, pick from {MODELS}")


def git_commit() -> str:
    try:
        run = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return run.stdout.strip()


def environment() -> dict:
    return {
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "machine": platform.platform(),
    }


def run_baselines(
    ds: Dataset,
    fold: int = 0,
    names: tuple[str, ...] = MODELS,
    seed: int = 0,
    options: dict[str, dict] | None = None,
) -> dict:
    """Metrics of each model on the validation and test sequences of one fold.

    Every model is scored on the same rows: the ones that have a full window of steps behind
    them, one out of EVAL_STRIDE. options holds keyword arguments for each fit function.
    """
    options = options or {}
    split = make_folds(ds.n_seq)[fold]
    train, val, test = ds.take(split.train), ds.take(split.val), ds.take(split.test)
    prep = Prepared.fit(train)
    scored = {
        "val": (val, window_rows(val, WINDOW, EVAL_STRIDE)),
        "test": (test, window_rows(test, WINDOW, EVAL_STRIDE)),
    }

    models = {}
    for name in names:
        start = time.time()
        model = fit_model(name, prep, train, seed, **options.get(name, {}))
        models[name] = {"fit_seconds": round(time.time() - start, 1)}
        for part, (d, rows) in scored.items():
            models[name][part] = point_metrics(d.health[rows], model.predict(d, rows))

    return {
        "fold": fold,
        "seed": seed,
        "sequences": {"train": len(train.ids), "val": len(val.ids), "test": len(test.ids)},
        "scored_rows": {part: len(rows) for part, (_, rows) in scored.items()},
        "environment": environment(),
        "models": models,
    }
