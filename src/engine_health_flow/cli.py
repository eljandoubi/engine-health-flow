import argparse
import json
from pathlib import Path

import numpy as np

from engine_health_flow import data, experiment, missing, report

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


def run(csv: Path, fold: int, names: tuple[str, ...], seed: int, out: Path) -> None:
    results = experiment.run_fold(data.read_csv(csv), fold, names, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    for name, res in results["models"].items():
        test = res["test"]
        pooled = test["pooled"]
        line = (
            f"{name:9s} test rmse {pooled['rmse']:.2e}  smape {pooled['smape']:.3f}  "
            f"pearson {pooled['pearson']:.3f}"
        )
        if "posterior" in test:
            post = test["posterior"]
            covered = np.mean(post["intervals"]["0.9"]["coverage"])
            line += f"  crps {post['crps_mean']:.2e}  90% interval covers {covered:.2f}"
        print(f"{line}  (fit {res['fit_seconds']:.0f}s)")
    print(f"written to {out}")


def lose_a_phase(csv: Path, fold: int, seed: int, out: Path) -> None:
    results = missing.run_missing(data.read_csv(csv), fold, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(report.missing_table([results]))
    print(f"written to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ehf", description="Turbofan health estimation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the benchmark csv and check its md5")
    p.add_argument("--dest", type=Path, default=DEFAULT_CSV)

    p = sub.add_parser("info", help="print a few numbers about the csv")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)

    p = sub.add_parser("run", help="fit models on one fold and write their metrics")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--models", default=",".join(experiment.MODELS), help="comma separated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="json file, default results/fold<k>.json")

    p = sub.add_parser("table", help="print markdown tables from the files written by run")
    p.add_argument("files", nargs="+", type=Path)

    p = sub.add_parser("missing", help="score the ukf and a flow when a flight phase is lost")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="json file, default results/missing_fold<k>.json")

    args = parser.parse_args(argv)
    if args.command == "download":
        print(data.download(args.dest))
    elif args.command == "info":
        info(args.csv)
    elif args.command == "run":
        names = tuple(args.models.split(","))
        unknown = set(names) - set(experiment.MODELS)
        if unknown:
            parser.error(f"unknown models: {', '.join(sorted(unknown))}")
        out = args.out or Path("results") / f"fold{args.fold}.json"
        run(args.csv, args.fold, names, args.seed, out)
    elif args.command == "table":
        results = report.load(args.files)
        print(report.main_table(results))
        print()
        print(report.indicator_table(results))
    elif args.command == "missing":
        out = args.out or Path("results") / f"missing_fold{args.fold}.json"
        lose_a_phase(args.csv, args.fold, args.seed, out)
