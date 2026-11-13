"""Plot the posterior of the ukf and of the hybrid flow on one test sequence.

    uv run --with matplotlib python scripts/plot_sequence.py --fold 0 --out docs/sequence.png

The bands are the central 90 % of the draws, after the spread of each model has been rescaled with
the factors fitted on the validation sequences of the same fold, as in ehf run.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import engine_health_flow  # noqa: E402
from engine_health_flow import data  # noqa: E402
from engine_health_flow.experiment import fit_model  # noqa: E402
from engine_health_flow.metrics import rescale_spread, spread_scales  # noqa: E402
from engine_health_flow.scaling import Prepared  # noqa: E402
from engine_health_flow.splits import make_folds  # noqa: E402
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows  # noqa: E402

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]
SHOWN = ["CmpFan_eff", "TrbL_eff", "TrbL_flow", "CmpBst_eff"]
MODELS = {"ukf": "#1f77b4", "hybrid": "#d95f02"}


def plot(csv: Path, fold: int, sequence: int, out: Path, options: dict | None = None) -> None:
    options = options or {}
    ds = data.read_csv(csv)
    split = make_folds(ds.n_seq)[fold]
    train, val, test = ds.take(split.train), ds.take(split.val), ds.take(split.test)
    prep = Prepared.fit(train)
    val_rows = window_rows(val, WINDOW, EVAL_STRIDE)

    one = test.take([sequence])
    rows = np.arange(WINDOW - 1, one.lengths[0])
    bands = {}
    for name in MODELS:
        model = fit_model(name, prep, train, val, 0, **options.get(name, {}))
        point, draws = model.posterior(val, val_rows)
        scales = spread_scales(val.health[val_rows], point, draws)
        mean, draws = model.posterior(one, rows)
        low, high = np.quantile(rescale_spread(mean, draws, scales), [0.05, 0.95], axis=1)
        bands[name] = (mean, low, high)

    fig, axes = plt.subplots(len(SHOWN), 1, figsize=(9, 2.2 * len(SHOWN)), sharex=True)
    maintenance = np.flatnonzero(one.maintenance)
    for ax, indicator in zip(axes, SHOWN, strict=True):
        k = data.HEALTH_SHORT.index(indicator)
        for at in maintenance:
            ax.axvline(at, color="0.85", lw=0.8)
        ax.plot(one.health[:, k], color="black", lw=1.6, label="truth")
        for name, color in MODELS.items():
            mean, low, high = bands[name]
            ax.fill_between(rows, low[:, k], high[:, k], color=color, alpha=0.25, lw=0)
            ax.plot(rows, mean[:, k], color=color, lw=0.9, label=f"{name} mean and 90 % band")
        ax.set_ylabel(indicator)
    axes[0].legend(loc="lower left", ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("row of the sequence (grey lines: maintenance)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw" / data.FILENAME)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--sequence", type=int, default=0, help="index among the test sequences")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/sequence.png")
    args = parser.parse_args()
    plot(args.csv, args.fold, args.sequence, args.out)


if __name__ == "__main__":
    main()
