"""Aggregate experiment metrics into a summary table."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def load_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sweep results.")
    parser.add_argument("--artifacts", default="./artifacts", help="Artifacts root")
    parser.add_argument("--top-k", type=int, default=10, help="Print top-k runs")
    args = parser.parse_args()

    rows: List[Dict[str, float | str]] = []
    for run_name in sorted(os.listdir(args.artifacts)):
        run_dir = os.path.join(args.artifacts, run_name)
        if not os.path.isdir(run_dir):
            continue
        val_path = os.path.join(run_dir, "val_metrics.json")
        test_path = os.path.join(run_dir, "test_metrics.json")
        if not (os.path.isfile(val_path) and os.path.isfile(test_path)):
            continue

        val_metrics = load_json(val_path)
        test_metrics = load_json(test_path)
        row: Dict[str, float | str] = {"run_name": run_name}
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "val_IC" in df.columns:
        df = df.sort_values("val_IC", ascending=False)

    summary_path = os.path.join(args.artifacts, "summary.csv")
    df.to_csv(summary_path, index=False)

    if args.top_k > 0 and not df.empty:
        print(df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
