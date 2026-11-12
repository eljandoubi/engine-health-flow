import argparse
from pathlib import Path

import numpy as np

from engine_health_flow import data

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ehf", description="Turbofan health estimation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the benchmark csv and check its md5")
    p.add_argument("--dest", type=Path, default=DEFAULT_CSV)

    p = sub.add_parser("info", help="print a few numbers about the csv")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)

    args = parser.parse_args(argv)
    if args.command == "download":
        print(data.download(args.dest))
    elif args.command == "info":
        info(args.csv)
