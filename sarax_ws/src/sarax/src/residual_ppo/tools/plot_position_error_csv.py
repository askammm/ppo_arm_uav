#!/usr/bin/env python3

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def load_csv(path: str) -> Dict[str, List[float]]:
    data = {
        "time_sec": [],
        "error_norm": [],
        "error_integral": [],
    }
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["time_sec"].append(float(row["time_sec"]))
            data["error_norm"].append(float(row["error_norm"]))
            data["error_integral"].append(float(row["error_integral"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot exported position error CSV files."
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to compare")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, defaults to CSV basenames",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.csv_files]
    if len(labels) != len(args.csv_files):
        raise SystemExit("Number of labels must match number of CSV files.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for path, label in zip(args.csv_files, labels):
        data = load_csv(path)
        axes[0].plot(data["time_sec"], data["error_norm"], label=label)
        axes[1].plot(data["time_sec"], data["error_integral"], label=label)

    axes[0].set_ylabel("Position Error Norm [m]")
    axes[0].set_title("Position Error Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Integral of Error Norm [m*s]")
    axes[1].set_title("Integrated Position Error Comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
