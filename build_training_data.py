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

BLOCK_LEN = 512


def discover_all_days(data_root: str) -> List[str]:
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


def process_skey(
    path: str,
    feature_cols: List[str],
    group_cols: Dict[str, List[str]],
    stats: Dict[str, Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=feature_cols + ["adjMidRet60s"])
    df = df.dropna(subset=["adjMidRet60s"])

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
    y = df["adjMidRet60s"].to_numpy(dtype=np.float32)
    return x, y


def build_tables(
    data_root: str,
    stats: Dict[str, Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor, List[dict], List[dict]]:
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

    x_segments: List[torch.Tensor] = []
    y_segments: List[torch.Tensor] = []
    segment_table: List[dict] = []
    block_table: List[dict] = []

    cursor = 0

    for day in discover_all_days(data_root):
        day_dir = os.path.join(data_root, day)
        skey_files = sorted(
            f for f in os.listdir(day_dir) if f.endswith(".parquet")
        )
        for skey_file in skey_files:
            skey = os.path.splitext(skey_file)[0]
            path = os.path.join(day_dir, skey_file)
            x_np, y_np = process_skey(path, feature_cols, group_cols, stats)
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

            if length >= BLOCK_LEN:
                for block_start in range(0, length - BLOCK_LEN + 1, BLOCK_LEN):
                    block_table.append(
                        {
                            "day": day,
                            "skey": skey,
                            "segment_start_idx": cursor,
                            "block_start": block_start,
                        }
                    )

            x_segments.append(x_tensor)
            y_segments.append(y_tensor)
            cursor += length

    x_all = torch.cat(x_segments, dim=0).to(dtype=torch.float32)
    y_all = torch.cat(y_segments, dim=0).to(dtype=torch.float32)
    return x_all, y_all, segment_table, block_table


def save_outputs(
    output_root: str,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    segment_table: List[dict],
    block_table: List[dict],
) -> None:
    os.makedirs(output_root, exist_ok=True)
    torch.save(x_all, os.path.join(output_root, "X_all.pt"))
    torch.save(y_all, os.path.join(output_root, "y_all.pt"))

    with open(os.path.join(output_root, "segment_table.pkl"), "wb") as f:
        pickle.dump(segment_table, f)
    with open(os.path.join(output_root, "block_table.pkl"), "wb") as f:
        pickle.dump(block_table, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline training data.")
    parser.add_argument("--data-root", default="./data", help="Raw data root.")
    parser.add_argument("--output-root", default="./global_data", help="Output root.")
    parser.add_argument("--stats-path", default="feature_stat.pt", help="Feature stats file.")
    args = parser.parse_args()

    stats = load_feature_stats(args.stats_path)
    x_all, y_all, segment_table, block_table = build_tables(args.data_root, stats)
    save_outputs(args.output_root, x_all, y_all, segment_table, block_table)


if __name__ == "__main__":
    main()
