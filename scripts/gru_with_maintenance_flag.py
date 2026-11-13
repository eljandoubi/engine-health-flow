"""Does the gru catch up with the ukf when it is told where the maintenance events are? One fold.

    uv run python scripts/gru_with_maintenance_flag.py --fold 0

In the main runs only the ukf reads the maintenance flag. Here the gru gets it as one more input
channel (1 on the row of the event, 0 elsewhere) and is trained exactly like the gru of `ehf run`.
The reference numbers are read from results/fold<k>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import engine_health_flow
from engine_health_flow import data
from engine_health_flow.neural import fit_gru
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]


class WithFlag:
    """Standardised sensors plus the maintenance flag. Stands in for Prepared."""

    def __init__(self, prep: Prepared):
        self.prep, self.health = prep, prep.health

    def x(self, ds: data.Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        flag = ds.maintenance[:, None].astype(np.float32)
        x = np.concatenate([self.prep.x(ds), flag], axis=1).astype(np.float32)
        return x if rows is None else x[rows]

    def y(self, ds: data.Dataset, rows: np.ndarray | None = None) -> np.ndarray:
        return self.prep.y(ds, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw" / data.FILENAME)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()

    ds = data.read_csv(args.csv)
    split = make_folds(ds.n_seq)[args.fold]
    train, test = ds.take(split.train), ds.take(split.test)
    prep = Prepared.fit(train)

    gru = fit_gru(WithFlag(prep), train)
    rows = window_rows(test, WINDOW, EVAL_STRIDE)
    error = np.sqrt(((test.health[rows] - gru.predict(test, rows)) ** 2).mean()) * 1e3
    print(f"gru with the maintenance flag, test RMSE x1e3: {error:.3f}")

    reference = json.loads((ROOT / "results" / f"fold{args.fold}.json").read_text())["models"]
    for name in ("gru", "ukf"):
        rmse = reference[name]["test"]["pooled"]["rmse"] * 1e3
        print(f"{name} in results/fold{args.fold}.json: {rmse:.3f}")


if __name__ == "__main__":
    main()
