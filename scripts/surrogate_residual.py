"""How close is the surrogate to the noise of the sensors? One fold.

    uv run python scripts/surrogate_residual.py --fold 0

Prints the residual of the surrogate on the training and validation sequences, the noise of the
sensors estimated from second differences, and the residual of a linear map for reference. All
numbers are standard deviations in standardised units, median over the sensors with the range.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

import engine_health_flow
from engine_health_flow import data
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.surrogate import fit_surrogate

ROOT = Path(engine_health_flow.__file__).resolve().parents[2]


def summary(values: np.ndarray) -> str:
    return f"{np.median(values):.4f} (min {values.min():.4f}, max {values.max():.4f})"


def second_difference_noise(ds: data.Dataset, x: np.ndarray) -> np.ndarray:
    """The second difference of a smooth signal is noise only, and it has 6 times the variance."""
    parts = [
        x[first + 2 : end] - 2 * x[first + 1 : end - 1] + x[first : end - 2]
        for first, end in zip(ds.starts[:-1], ds.starts[1:], strict=True)
    ]
    return np.concatenate(parts).std(0) / np.sqrt(6)


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

    for name, part in (("train", train), ("validation", val)):
        health = torch.from_numpy(prep.y(part)).double()
        with torch.no_grad():
            residual = prep.x(part) - surrogate(health).numpy()
        print(f"surrogate residual, {name} sequences: {summary(residual.std(0))}")
    noise = second_difference_noise(train, prep.x(train))
    print(f"sensor noise from second differences:  {summary(noise)}")

    linear = LinearRegression().fit(prep.y(train), prep.x(train))
    residual = prep.x(val) - linear.predict(prep.y(val))
    print(f"linear map residual, validation:       {summary(residual.std(0))}")


if __name__ == "__main__":
    main()
