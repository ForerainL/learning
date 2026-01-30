"""Offline preprocessing to build per-day training tensors.

This script normalizes and saves tensors once so training stays fast:
no pandas or normalization during model training.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def discover_all_days(data_root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(data_root) if d.isdigit() and len(d) == 8
    )


def load_feature_stats(path: str) -> Dict[str, Dict[str, object]]:
    stats = torch.load(path)
    if "heavy_zero" in stats:
        return stats
    if "heavy" in stats:
        stats["heavy_zero"] = stats["heavy"]
    return stats


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.array(value)


def normalize_smooth(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    mean = _to_numpy(stats.get("mean", 0.0)).astype(np.float32)
    std = _to_numpy(stats.get("std", 1.0)).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    normalized = (x - mean) / std
    clip = stats.get("clip")
    if clip is not None:
        clip_arr = _to_numpy(clip).astype(np.float32)
        normalized = np.clip(normalized, -clip_arr, clip_arr)
    return normalized


def normalize_heavy_zero(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    normalized = x.copy()
    nonzero = normalized != 0

    if stats.get("log1p", False):
        normalized[nonzero] = np.log1p(normalized[nonzero])

    scale = stats.get("scale")
    if scale is not None:
        scale_arr = _to_numpy(scale).astype(np.float32)
        scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
        normalized = normalized / scale_arr
    return normalized


def normalize_discrete(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    if "min" in stats and "max" in stats:
        min_val = _to_numpy(stats.get("min")).astype(np.float32)
        max_val = _to_numpy(stats.get("max")).astype(np.float32)
        denom = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
        return (x - min_val) / denom

    scale = stats.get("scale") or stats.get("max")
    if scale is None:
        return x
    scale_arr = _to_numpy(scale).astype(np.float32)
    scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
    return x / scale_arr


def process_skey(
    path: str,
    feature_cols: List[str],
    smooth_cols: List[str],
    heavy_cols: List[str],
    discrete_cols: List[str],
    stats: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=feature_cols + ["y"])
    df = df.dropna(subset=["y"])

    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0)

    xs = df[smooth_cols].to_numpy(dtype=np.float32) if smooth_cols else np.empty((len(df), 0), dtype=np.float32)
    xh = df[heavy_cols].to_numpy(dtype=np.float32) if heavy_cols else np.empty((len(df), 0), dtype=np.float32)
    xd = df[discrete_cols].to_numpy(dtype=np.float32) if discrete_cols else np.empty((len(df), 0), dtype=np.float32)

    if smooth_cols:
        xs = normalize_smooth(xs, stats["smooth"])
    if heavy_cols:
        xh = normalize_heavy_zero(xh, stats["heavy_zero"])
    if discrete_cols:
        xd = normalize_discrete(xd, stats["discrete"])

    x = np.concatenate([xs, xh, xd], axis=1).astype(np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    return x, y


def process_day(
    day: str,
    data_root: str,
    output_root: str,
    stats: Dict[str, Dict[str, object]],
) -> None:
    day_dir = os.path.join(data_root, day)
    skey_files = sorted(
        f for f in os.listdir(day_dir) if f.endswith(".parquet")
    )
    if not skey_files:
        return

    smooth_cols = list(stats["smooth"]["cols"])
    heavy_cols = list(stats["heavy_zero"]["cols"])
    discrete_cols = list(stats["discrete"]["cols"])
    feature_cols = smooth_cols + heavy_cols + discrete_cols

    x_list = []
    y_list = []
    length = None

    for skey_file in skey_files:
        path = os.path.join(day_dir, skey_file)
        x, y = process_skey(
            path,
            feature_cols,
            smooth_cols,
            heavy_cols,
            discrete_cols,
            stats,
        )
        if length is None:
            length = len(y)
        elif len(y) != length:
            raise ValueError(
                f"Inconsistent length for {day}/{skey_file}: {len(y)} vs {length}"
            )
        x_list.append(torch.from_numpy(x))
        y_list.append(torch.from_numpy(y))

    if length is None:
        return

    x_day = torch.stack(x_list, dim=0).to(dtype=torch.float32)
    y_day = torch.stack(y_list, dim=0).to(dtype=torch.float32)

    out_dir = os.path.join(output_root, day)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(x_day, os.path.join(out_dir, "X.pt"))
    torch.save(y_day, os.path.join(out_dir, "y.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline training tensors.")
    parser.add_argument("--data-root", default="./data", help="Raw data root.")
    parser.add_argument("--output-root", default="./tensor_data", help="Output root.")
    parser.add_argument("--stats-path", default="feature_stat.pt", help="Feature stats file.")
    args = parser.parse_args()

    stats = load_feature_stats(args.stats_path)

    for day in discover_all_days(args.data_root):
        process_day(day, args.data_root, args.output_root, stats)


if __name__ == "__main__":
    main()
