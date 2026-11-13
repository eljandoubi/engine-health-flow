"""Test error of the ukf and the gru by the number of rows since the last maintenance, on one fold.

    uv run python scripts/error_by_time_since_maintenance.py --fold 0

The hard indicators are the three that every model gets worst: fan efficiency and the two low
pressure turbine ones. The script also prints the ukf error on the training sequences, which its
surrogate has seen, next to the error on new sequences.
"""

import argparse
from pathlib import Path

import numpy as np

import engine_health_flow
from engine_health_flow import data
from engine_health_flow.neural import fit_gru
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.ukf import fit_ukf
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]
HARD = ["CmpFan_eff", "TrbL_eff", "TrbL_flow"]
BINS = [(0, 50), (50, 100), (100, 200), (200, 300), (300, None)]


def rows_since_maintenance(ds: data.Dataset) -> np.ndarray:
    """For every row, how many rows ago the last maintenance or the start of its sequence was."""
    since = np.zeros(len(ds.health), dtype=int)
    for first, end in zip(ds.starts[:-1], ds.starts[1:], strict=True):
        count = 0
        for row in range(first, end):
            count = 0 if ds.maintenance[row] else count + 1
            since[row] = count
    return since


def rmse(truth: np.ndarray, pred: np.ndarray, columns: list[int]) -> float:
    return float(np.sqrt(((truth[:, columns] - pred[:, columns]) ** 2).mean()) * 1e3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw" / data.FILENAME)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()

    ds = data.read_csv(args.csv)
    split = make_folds(ds.n_seq)[args.fold]
    train, val, test = ds.take(split.train), ds.take(split.val), ds.take(split.test)
    prep = Prepared.fit(train)

    ukf = fit_ukf(prep, train, val)
    r_scale = float((ukf.obs_cov[0, 0] / ukf.surrogate.noise[0] ** 2) ** 0.5)
    print(f"ukf noise scales: process {ukf.q_scale}, measurement {r_scale:.1f}", flush=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        rows = window_rows(part, WINDOW, EVAL_STRIDE)
        everything = list(range(part.health.shape[1]))
        error = rmse(part.health[rows], ukf.predict(part, rows), everything)
        print(f"ukf RMSE x1e3 on the {name} sequences: {error:.3f}", flush=True)

    gru = fit_gru(prep, train)
    rows = window_rows(test, WINDOW, EVAL_STRIDE)
    truth = test.health[rows]
    pred = {"ukf": ukf.predict(test, rows), "gru": gru.predict(test, rows)}
    since = rows_since_maintenance(test)[rows]
    hard = [data.HEALTH_SHORT.index(name) for name in HARD]
    easy = [i for i in range(truth.shape[1]) if i not in hard]

    print("\nTest RMSE x1e3, ukf / gru, by rows since the last maintenance")
    print(f"{'rows since':>12} {'scored rows':>12} {'3 hard indicators':>20} {'7 others':>14}")
    for low, high in BINS:
        mask = (since >= low) & ((since < high) if high else True)
        cells = []
        for columns in (hard, easy):
            both = [rmse(truth[mask], pred[name][mask], columns) for name in ("ukf", "gru")]
            cells.append(f"{both[0]:.2f} / {both[1]:.2f}")
        label = f"{low} to {high}" if high else f"{low} and more"
        print(f"{label:>12} {int(mask.sum()):>12} {cells[0]:>20} {cells[1]:>14}")


if __name__ == "__main__":
    main()
