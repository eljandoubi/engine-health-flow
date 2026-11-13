"""Validation RMSE of the ukf over a wider grid of its two noise scales. One fold.

    uv run python scripts/ukf_noise_grid.py --fold 0

fit_ukf picks the best cell of a small grid. This prints the whole surface on a wider one, to see
whether the pick sits on a plateau or at the edge of a slope. Rows are the process noise scale
(times the covariance of the steps), columns the measurement noise scale (times the surrogate
residual). The numbers are the RMSE x1e3 on the validation rows.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import engine_health_flow
from engine_health_flow import data
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.surrogate import fit_surrogate
from engine_health_flow.ukf import UKFModel, estimate_dynamics
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]
PROCESS = (4.0, 16.0, 64.0, 256.0, 1024.0)
MEASUREMENT = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw" / data.FILENAME)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()

    ds = data.read_csv(args.csv)
    split = make_folds(ds.n_seq)[args.fold]
    train, val = ds.take(split.train), ds.take(split.val)
    prep = Prepared.fit(train)
    surrogate = fit_surrogate(prep, train)
    dyn = estimate_dynamics(prep, train)
    base = torch.diag(torch.from_numpy(surrogate.noise.astype(np.float64)) ** 2)
    rows = window_rows(val, WINDOW, EVAL_STRIDE)

    print("validation RMSE x1e3, process scale (rows) by measurement scale (columns)")
    print("process".rjust(8) + "".join(f"{r:8.1f}" for r in MEASUREMENT))
    for q in PROCESS:
        errors = []
        for r in MEASUREMENT:
            model = UKFModel(prep, surrogate, dyn, base * r**2, q)
            error = np.sqrt(((val.health[rows] - model.predict(val, rows)) ** 2).mean())
            errors.append(error * 1e3)
        print(f"{q:8.0f}" + "".join(f"{e:8.3f}" for e in errors), flush=True)


if __name__ == "__main__":
    main()
