"""Does a much longer training change the scores of the MLP and the GRU baselines? One fold.

    uv run python scripts/training_budget.py --fold 0

ehf run trains the MLP for 4000 steps and the GRU for 3000. Here they get 20000 and 12000.
"""

import argparse
from pathlib import Path

import engine_health_flow
from engine_health_flow import data
from engine_health_flow.metrics import point_metrics
from engine_health_flow.neural import fit_gru, fit_mlp
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.windows import EVAL_STRIDE, WINDOW, window_rows

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]
LONG = {"mlp": (fit_mlp, 20000), "gru": (fit_gru, 12000)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw" / data.FILENAME)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    args = parser.parse_args()

    ds = data.read_csv(args.csv)
    split = make_folds(ds.n_seq)[args.fold]
    train, val, test = ds.take(split.train), ds.take(split.val), ds.take(split.test)
    prep = Prepared.fit(train)

    for name, (fit, steps) in LONG.items():
        model = fit(prep, train, steps=steps)
        for part, d in (("val", val), ("test", test)):
            rows = window_rows(d, WINDOW, EVAL_STRIDE)
            pooled = point_metrics(d.health[rows], model.predict(d, rows))["pooled"]
            print(
                f"{name} {steps} steps, {part}: RMSE x1e3 {pooled['rmse'] * 1e3:.3f}  "
                f"SMAPE {pooled['smape']:.3f}  Pearson {pooled['pearson']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
