"""Fit the models on one fold and collect their metrics."""

import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from engine_health_flow.baselines import fit_boosting, fit_ridge
from engine_health_flow.data import Dataset
from engine_health_flow.flow import fit_flow
from engine_health_flow.hybrid import fit_hybrid
from engine_health_flow.metrics import (
    point_metrics,
    posterior_metrics,
    rescale_spread,
    spread_scales,
)
from engine_health_flow.neural import fit_gru, fit_mlp
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.ukf import fit_ukf
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows

MODELS = ("ridge", "boosting", "mlp", "gru", "ukf", "flow", "hybrid")


def fit_model(name: str, prep: Prepared, train: Dataset, val: Dataset, seed: int, **options):
    """Only the ukf and the hybrid look at val: the ukf picks its two noise scales there."""
    if name == "ridge":
        return fit_ridge(prep, train, **options)
    if name == "boosting":
        return fit_boosting(prep, train, seed=seed, **options)
    if name == "mlp":
        return fit_mlp(prep, train, seed=seed, **options)
    if name == "gru":
        return fit_gru(prep, train, seed=seed, **options)
    if name == "ukf":
        return fit_ukf(prep, train, val, seed=seed, **options)
    if name == "flow":
        return fit_flow(prep, train, seed=seed, **options)
    if name == "hybrid":
        return fit_hybrid(prep, train, val, seed=seed, **options)
    raise ValueError(f"unknown model {name!r}, pick from {MODELS}")


def git_commit(folder: Path = Path(__file__).parent) -> str:
    """Short hash of HEAD, followed by -dirty when tracked files have uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("-dirty" if changed.stdout.strip() else "")


def environment() -> dict:
    return {
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "machine": platform.platform(),
    }


def score_model(model, scored: dict) -> dict:
    """Metrics of a fitted model on each part of scored, the first part being validation.

    A model with a posterior is scored on its draws too, and again with the spread of the draws
    rescaled by factors fitted on validation. Those rescaled numbers are only held out on test.
    """
    result, scales = {}, None
    for part, (d, rows) in scored.items():
        truth = d.health[rows]
        if not hasattr(model, "posterior"):
            result[part] = point_metrics(truth, model.predict(d, rows))
            continue
        point, draws = model.posterior(d, rows)
        if scales is None:
            scales = spread_scales(truth, point, draws)
        result[part] = point_metrics(truth, point)
        result[part]["posterior"] = posterior_metrics(truth, draws)
        rescaled = rescale_spread(point, draws, scales)
        result[part]["posterior_rescaled"] = posterior_metrics(truth, rescaled)
    if scales is not None:
        result["spread_scales"] = scales.tolist()
    return result


def run_fold(
    ds: Dataset,
    fold: int = 0,
    names: tuple[str, ...] = MODELS,
    seed: int = 0,
    options: dict[str, dict] | None = None,
) -> dict:
    """Metrics of each model on the validation and test sequences of one fold.

    Every model is scored on the same rows: the ones that have a full window of steps behind
    them, one out of EVAL_STRIDE. options holds keyword arguments for each fit function. Models
    that can draw from the posterior (the ukf and the flow) are scored on the draws too.
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
        start = time.monotonic()  # unlike time.time() it does not count a sleeping laptop
        model = fit_model(name, prep, train, val, seed, **options.get(name, {}))
        models[name] = {"fit_seconds": round(time.monotonic() - start, 1)}
        models[name].update(score_model(model, scored))
        print(f"{name}: done after {time.monotonic() - start:.0f}s", flush=True)

    return {
        "fold": fold,
        "seed": seed,
        "sequences": {"train": len(train.ids), "val": len(val.ids), "test": len(test.ids)},
        "scored_rows": {part: len(rows) for part, (_, rows) in scored.items()},
        "environment": environment(),
        "models": models,
    }
