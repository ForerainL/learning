"""Offline normalization to build global tensors and block metadata.

This script is intentionally slow and run once. It produces:
- X_all.pt / y_all.pt: contiguous tensors for fast training
- segment_table.pkl / block_table.pkl: boundaries for safe sampling
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch



def discover_all_days(data_root: str) -> List[str]:
    return sorted(d for d in os.listdir(data_root) if d.isdigit() and len(d) == 8)


def load_feature_stats(path: str) -> Dict[str, Dict[str, object]]:
    return torch.load(path)


def apply_feature_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    stats: Dict[str, Dict[str, object]],
) -> np.ndarray:
    columns = []
    for col in feature_cols:
        series = df[col].astype(float)
        spec = stats[col]
        clip_low = spec["clip_low"]
        clip_high = spec["clip_high"]
        clipped = series.clip(lower=clip_low, upper=clip_high)
        ftype = spec["type"]

        if ftype == "zscore":
            mean = spec["mean"]
            std = spec["std"]
            values = (clipped - mean) / (std if std != 0 else 1.0)
            values = np.clip(values, -5.0, 5.0)
        elif ftype == "right_skew":
            clipped = clipped.clip(lower=0.0)
            logged = np.log1p(clipped)
            log_mean = spec["log_mean"]
            log_std = spec["log_std"]
            values = (logged - log_mean) / (log_std if log_std != 0 else 1.0)
            values = np.clip(values, -5.0, 5.0)
        else:
            max_abs = spec["max_abs"]
            values = clipped / (max_abs if max_abs != 0 else 1.0)
            values = np.clip(values, -5.0, 5.0)

        columns.append(values.to_numpy(dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)


def process_skey(
    path: str,
    feature_cols: List[str],
    stats: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=feature_cols + ["adjMidRet60s"])
    df = df.dropna(subset=["adjMidRet60s"])

    x = apply_feature_stats(df, feature_cols, stats)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = (df["adjMidRet60s"] * 100.0).to_numpy(dtype=np.float32)
    return x, y


def build_tables(
    data_root: str,
    stats: Dict[str, Dict[str, object]],
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict], List[int]]:
    feature_cols = [f"x_{i:03d}" for i in range(137)]

    x_segments: List[torch.Tensor] = []
    y_segments: List[torch.Tensor] = []
    segment_table: List[dict] = []
    index_table: List[int] = []

    cursor = 0

    for day in discover_all_days(data_root):
        day_dir = os.path.join(data_root, day)
        skey_files = sorted(
            f for f in os.listdir(day_dir) if f.endswith(".parquet")
        )
        for skey_file in skey_files:
            skey = os.path.splitext(skey_file)[0]
            path = os.path.join(day_dir, skey_file)
            x_np, y_np = process_skey(path, feature_cols, stats)
            length = len(y_np)
            if length == 0:
                continue

            x_tensor = torch.from_numpy(x_np)
            y_tensor = torch.from_numpy(y_np)

            segment_table.append(
                {
                    "day": day,
                    "skey": skey,
                    "start_idx": cursor,
                    "length": length,
                }
            )

            if length > window:
                start = cursor + window
                end = cursor + length
                index_table.extend(range(start, end))

            x_segments.append(x_tensor)
            y_segments.append(y_tensor)
            cursor += length

    x_all = torch.cat(x_segments, dim=0).to(dtype=torch.float32)
    y_all = torch.cat(y_segments, dim=0).to(dtype=torch.float32)
    return x_all, y_all, segment_table, index_table


def save_outputs(
    output_root: str,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    segment_table: List[dict],
    index_table: List[int],
) -> None:
    os.makedirs(output_root, exist_ok=True)
    torch.save(x_all, os.path.join(output_root, "X_all.pt"))
    torch.save(y_all, os.path.join(output_root, "y_all.pt"))

    with open(os.path.join(output_root, "segment_table.pkl"), "wb") as f:
        pickle.dump(segment_table, f)
    with open(os.path.join(output_root, "index_table.pkl"), "wb") as f:
        pickle.dump(index_table, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline training data.")
    parser.add_argument("--data-root", default="./data", help="Raw data root.")
    parser.add_argument("--output-root", default="./global_data", help="Output root.")
    parser.add_argument("--stats-path", default="feature_stat.pt", help="Feature stats file.")
    parser.add_argument("--window", type=int, default=64, help="Window length.")
    args = parser.parse_args()

    stats = load_feature_stats(args.stats_path)
    x_all, y_all, segment_table, index_table = build_tables(args.data_root, stats, args.window)
    save_outputs(args.output_root, x_all, y_all, segment_table, index_table)


if __name__ == "__main__":
    main()
