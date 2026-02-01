"""Offline preprocessing to save per-(day,skey) tensors and index_table.

Output format per file:
  output_root/YYYYMMDD/SKEY.pt  -> {"X": Tensor[Ti,F], "y": Tensor[Ti]}

index_table.pkl contains (day, skey, t_end) for all valid windows.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def discover_days(data_root: str) -> List[str]:
    return sorted(d for d in os.listdir(data_root) if d.isdigit() and len(d) == 8)


def load_feature_stats(path: str) -> Dict[str, Dict[str, object]]:
    return torch.load(path)


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


def normalize_heavy_tail(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    normalized = x.astype(np.float32)
    if stats.get("log1p", False):
        normalized = np.log1p(np.clip(normalized, a_min=0, a_max=None))

    if "median" in stats and "mad" in stats:
        median = _to_numpy(stats.get("median")).astype(np.float32)
        mad = _to_numpy(stats.get("mad")).astype(np.float32)
        mad = np.where(mad == 0, 1.0, mad)
        normalized = (normalized - median) / mad

    scale = stats.get("scale")
    if scale is not None:
        scale_arr = _to_numpy(scale).astype(np.float32)
        scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
        normalized = normalized / scale_arr

    clip = stats.get("clip")
    if clip is not None:
        clip_arr = _to_numpy(clip).astype(np.float32)
        normalized = np.clip(normalized, -clip_arr, clip_arr)
    return normalized


def normalize_discrete(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    if "min" in stats and "max" in stats:
        min_val = _to_numpy(stats.get("min")).astype(np.float32)
        max_val = _to_numpy(stats.get("max")).astype(np.float32)
        denom = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
        return (x - min_val) / denom

    scale = stats.get("scale") or stats.get("max")
    if scale is None:
        return x.astype(np.float32)
    scale_arr = _to_numpy(scale).astype(np.float32)
    scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
    return x / scale_arr


def normalize_zero_dominant(x: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    normalized = x.astype(np.float32)
    nonzero = normalized != 0

    if stats.get("log1p", False):
        normalized[nonzero] = np.log1p(normalized[nonzero])

    scale = stats.get("scale")
    if scale is not None:
        scale_arr = _to_numpy(scale).astype(np.float32)
        scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)
        normalized[nonzero] = normalized[nonzero] / scale_arr

    return normalized


def process_one_file(
    path: str,
    feature_cols: List[str],
    group_cols: Dict[str, List[str]],
    stats: Dict[str, Dict[str, object]],
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=feature_cols + [label_col])
    df = df.dropna(subset=[label_col])

    df[feature_cols] = df[feature_cols].ffill()

    xs = df[group_cols["smooth"]].to_numpy(dtype=np.float32) if group_cols["smooth"] else np.empty((len(df), 0), dtype=np.float32)
    xh = df[group_cols["heavy_tail"]].to_numpy(dtype=np.float32) if group_cols["heavy_tail"] else np.empty((len(df), 0), dtype=np.float32)
    xd = df[group_cols["discrete"]].to_numpy(dtype=np.float32) if group_cols["discrete"] else np.empty((len(df), 0), dtype=np.float32)
    xz = df[group_cols["zero_dominant"]].to_numpy(dtype=np.float32) if group_cols["zero_dominant"] else np.empty((len(df), 0), dtype=np.float32)

    if group_cols["smooth"]:
        xs = normalize_smooth(xs, stats["smooth"])
    if group_cols["heavy_tail"]:
        xh = normalize_heavy_tail(xh, stats["heavy_tail"])
    if group_cols["discrete"]:
        xd = normalize_discrete(xd, stats["discrete"])
    if group_cols["zero_dominant"]:
        xz = normalize_zero_dominant(xz, stats["zero_dominant"])

    x = np.concatenate([xs, xh, xd, xz], axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[label_col].to_numpy(dtype=np.float32)
    return x, y


def build_index_table(
    data_root: str,
    output_root: str,
    days: List[str],
    feature_cols: List[str],
    group_cols: Dict[str, List[str]],
    stats: Dict[str, Dict[str, object]],
    label_col: str,
    window: int,
) -> List[Tuple[str, str, int]]:
    index_table: List[Tuple[str, str, int]] = []

    for day in days:
        day_dir = os.path.join(output_root, day)
        os.makedirs(day_dir, exist_ok=True)

        raw_day_dir = os.path.join(data_root, day)
        skey_files = sorted(f for f in os.listdir(raw_day_dir) if f.endswith(".parquet"))
        for skey_file in skey_files:
            skey = os.path.splitext(skey_file)[0]
            path = os.path.join(raw_day_dir, skey_file)
            x_np, y_np = process_one_file(path, feature_cols, group_cols, stats, label_col)
            if len(y_np) == 0:
                continue
            payload = {"X": torch.from_numpy(x_np), "y": torch.from_numpy(y_np)}
            torch.save(payload, os.path.join(day_dir, f"{skey}.pt"))

            for t_end in range(window, len(y_np)):
                index_table.append((day, skey, t_end))

    return index_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-(day,skey) tensors and index table.")
    parser.add_argument("--data-root", default="./data", help="Raw data root.")
    parser.add_argument("--output-root", default="./tensor_data", help="Output root.")
    parser.add_argument("--stats-path", default="feature_stat.pt", help="Feature stats file.")
    parser.add_argument("--label-col", default="adjMidRet60s", help="Label column name.")
    parser.add_argument("--window", type=int, default=64, help="Window length.")
    args = parser.parse_args()

    stats = load_feature_stats(args.stats_path)
    smooth_cols = list(stats["smooth"]["cols"])
    heavy_cols = list(stats["heavy_tail"]["cols"])
    discrete_cols = list(stats["discrete"]["cols"])
    zero_cols = list(stats["zero_dominant"]["cols"])

    feature_cols = smooth_cols + heavy_cols + discrete_cols + zero_cols
    group_cols = {
        "smooth": smooth_cols,
        "heavy_tail": heavy_cols,
        "discrete": discrete_cols,
        "zero_dominant": zero_cols,
    }

    days = discover_days(args.data_root)
    index_table = build_index_table(
        args.data_root,
        args.output_root,
        days,
        feature_cols,
        group_cols,
        stats,
        args.label_col,
        args.window,
    )

    with open(os.path.join(args.output_root, "index_table.pkl"), "wb") as f:
        pickle.dump(index_table, f)


if __name__ == "__main__":
    main()
