import argparse
import json
from pathlib import Path

import numpy as np

from engine_health_flow import data, experiment

DEFAULT_CSV = Path("data/raw") / data.FILENAME


def info(csv: Path) -> None:
    ds = data.read_csv(csv)
    n = ds.lengths
    print(f"{ds.n_seq} sequences, {len(ds.health)} rows")
    print(f"length {n.min()} to {n.max()} (median {int(np.median(n))})")
    print(f"{int(ds.maintenance.sum())} maintenance events")
    print("health indicators (min, max):")
    for name, col in zip(data.HEALTH, ds.health.T, strict=True):
        print(f"  {name:24s} {col.min():+.4f} {col.max():+.4f}")


def baselines(csv: Path, fold: int, names: tuple[str, ...], seed: int, out: Path) -> None:
    results = experiment.run_baselines(data.read_csv(csv), fold, names, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    for name, res in results["models"].items():
        test = res["test"]["pooled"]
        print(
            f"{name:9s} test rmse {test['rmse']:.2e}  smape {test['smape']:.3f}  "
            f"pearson {test['pearson']:.3f}  (fit {res['fit_seconds']:.0f}s)"
        )
    print(f"written to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ehf", description="Turbofan health estimation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the benchmark csv and check its md5")
    p.add_argument("--dest", type=Path, default=DEFAULT_CSV)

    p = sub.add_parser("info", help="print a few numbers about the csv")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)

    p = sub.add_parser("baselines", help="fit the baselines on one fold and write the metrics")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--models", default=",".join(experiment.MODELS), help="comma separated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="json file, default results/fold<k>.json")

    args = parser.parse_args(argv)
    if args.command == "download":
        print(data.download(args.dest))
    elif args.command == "info":
        info(args.csv)
    elif args.command == "baselines":
        names = tuple(args.models.split(","))
        unknown = set(names) - set(experiment.MODELS)
        if unknown:
            parser.error(f"unknown models: {', '.join(sorted(unknown))}")
        out = args.out or Path("results") / f"fold{args.fold}.json"
        baselines(args.csv, args.fold, names, args.seed, out)
