"""Turn the json files written by ehf run into markdown tables."""

import json
from pathlib import Path

import numpy as np

from engine_health_flow.data import HEALTH_SHORT


def load(paths: list[Path]) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def _dig(model: dict, *keys: str):
    """A nested value of the test metrics, or None when that model does not have it."""
    value = model["test"]
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _coverage(model: dict, kind: str):
    levels = _dig(model, kind, "intervals", "0.9")
    return None if levels is None else float(np.mean(levels["coverage"]))


def _crps(model: dict, kind: str):
    value = _dig(model, kind, "crps_mean")
    return None if value is None else value * 1e3


def _rmse(model: dict):
    return _dig(model, "pooled", "rmse") * 1e3


def _raw_crps(model: dict):
    return _crps(model, "posterior")


def _rescaled_crps(model: dict):
    return _crps(model, "posterior_rescaled")


COLUMNS = [
    ("RMSE x1e3", _rmse, 2),
    ("SMAPE", lambda m: _dig(m, "pooled", "smape"), 3),
    ("Pearson", lambda m: _dig(m, "pooled", "pearson"), 3),
    ("CRPS x1e3", _raw_crps, 3),
    ("90% cover", lambda m: _coverage(m, "posterior"), 2),
    ("CRPS x1e3 rescaled", _rescaled_crps, 3),
    ("90% cover rescaled", lambda m: _coverage(m, "posterior_rescaled"), 2),
]
PAIRED_METRICS = [(_rmse, 2), (_raw_crps, 3), (_rescaled_crps, 3)]


def _cell(values: list, digits: int) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = f"{np.mean(values):.{digits}f}"
    if len(values) == 1:
        return mean
    return f"{mean} ± {np.std(values, ddof=1):.{digits}f}"


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main_table(results: list[dict]) -> str:
    """One row per model with the test metrics, mean ± std over the files when there are several."""
    rows = []
    for name in results[0]["models"]:
        models = [r["models"][name] for r in results if name in r["models"]]
        rows.append([name] + [_cell([get(m) for m in models], d) for _, get, d in COLUMNS])
    return _markdown(["model"] + [title for title, _, _ in COLUMNS], rows)


PAIRS = (("hybrid", "ukf"), ("flow", "gru"), ("flow", "ukf"), ("hybrid", "flow"))


def _difference(results: list[dict], first: str, second: str, get, digits: int) -> str:
    both = [(get(r["models"][first]), get(r["models"][second])) for r in results]
    diffs = [a - b for a, b in both if a is not None and b is not None]
    if not diffs:
        return "-"
    lower = sum(d < 0 for d in diffs)
    return f"{np.mean(diffs):+.{digits}f} (lower in {lower} of {len(diffs)})"


def paired_table(results: list[dict], pairs=PAIRS) -> str:
    """First model minus second, fold by fold: the mean difference and how often it is below 0."""
    rows = []
    for first, second in pairs:
        if first in results[0]["models"] and second in results[0]["models"]:
            cells = [_difference(results, first, second, get, d) for get, d in PAIRED_METRICS]
            rows.append([f"{first} - {second}", *cells])
    return _markdown(["pair", "RMSE x1e3", "CRPS x1e3", "CRPS x1e3 rescaled"], rows)


def indicator_table(results: list[dict]) -> str:
    """Test RMSE x1e3 of every health indicator, by model."""
    rows = []
    for name in results[0]["models"]:
        per = np.array([r["models"][name]["test"]["rmse"] for r in results if name in r["models"]])
        rows.append([name] + [f"{v:.2f}" for v in per.mean(0) * 1e3])
    return _markdown(["model"] + HEALTH_SHORT, rows)


MISSING_COLUMNS = [
    ("RMSE x1e3", lambda s: s["rmse"] * 1e3, 2),
    ("CRPS x1e3 rescaled", lambda s: s["rescaled"]["crps_mean"] * 1e3, 3),
    ("90% cover", lambda s: s["coverage_90"], 2),
    ("90% cover rescaled", lambda s: s["rescaled"]["coverage_90"], 2),
]


def missing_table(results: list[dict]) -> str:
    """Test scores of the ukf and the flow for each lost flight phase."""
    models = ("ukf", "flow")
    header = ["lost phase"] + [f"{title} {m}" for title, _, _ in MISSING_COLUMNS for m in models]
    rows = []
    for condition in results[0]["conditions"]:
        cells = [condition]
        for _, get, digits in MISSING_COLUMNS:
            for model in models:
                values = [get(r["conditions"][condition][model]) for r in results]
                cells.append(_cell(values, digits))
        rows.append(cells)
    return _markdown(header, rows)
