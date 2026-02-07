"""Analysis utilities for aligning predictions with market data."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _segment_y_slice(
    segment_table: Sequence[Mapping[str, object]],
    day: str,
    skey: str,
    window: int,
) -> tuple[int, int, int]:
    if window <= 0:
        raise ValueError("window must be positive.")
    y_start = 0
    match_index = None
    y_len = None
    for idx, seg in enumerate(segment_table):
        seg_day = str(seg["day"])
        seg_skey = str(seg["skey"])
        seg_len = int(seg["length"])
        seg_y_len = seg_len - window
        if seg_day == day and seg_skey == skey:
            match_index = idx
            y_len = seg_y_len
            break
        y_start += max(seg_y_len, 0)
    if match_index is None or y_len is None:
        raise ValueError(f"No segment found for day={day} skey={skey}.")
    if y_len <= 0:
        raise ValueError(
            f"Segment day={day} skey={skey} has no valid labels: length={seg_len}, window={window}."
        )
    y_end = y_start + y_len
    return match_index, y_start, y_end


def build_day_skey_md_alpha_df(
    md_folder: str,
    x_parquet_folder: str,
    day: str,
    skey: str,
    segment_table: list[dict],
    y_true: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    window: int = 64,
) -> pd.DataFrame:
    """Build a merged MD + alpha DataFrame for a single (day, skey).

    This follows the same test_index ordering as train_optim.build_index_from_segments
    and the BlockDataset windowing logic: for each segment, the first `window`
    ticks are discarded, and remaining labels are concatenated in segment_table order.
    """
    if not isinstance(md_folder, str) or not md_folder:
        raise ValueError("md_folder must be a non-empty string.")
    if not isinstance(x_parquet_folder, str) or not x_parquet_folder:
        raise ValueError("x_parquet_folder must be a non-empty string.")
    if not isinstance(day, str) or not day:
        raise ValueError("day must be a non-empty string.")
    if not isinstance(skey, str) or not skey:
        raise ValueError("skey must be a non-empty string.")

    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array.")

    _, y_start, y_end = _segment_y_slice(segment_table, day, skey, window)
    if y_end > len(y_true_arr):
        raise ValueError(
            f"y_true slice exceeds array length: end={y_end}, len={len(y_true_arr)}."
        )
    y_slice = y_true_arr[y_start:y_end]

    x_parquet_path = os.path.join(x_parquet_folder, day, f"{skey}.parquet")
    x_df = pd.read_parquet(x_parquet_path, columns=["obe_seq_num", "y"])
    x_df = x_df.dropna(subset=["y"]).reset_index(drop=True)

    if len(x_df) != len(y_slice):
        raise ValueError(
            "Filtered x.parquet row count does not match y slice length: "
            f"{len(x_df)} vs {len(y_slice)}."
        )
    if not np.allclose(x_df["y"].to_numpy(), y_slice, equal_nan=False):
        raise ValueError("x.parquet y values do not align with y_true slice.")

    alpha_df = pd.DataFrame({"obe_seq_num": x_df["obe_seq_num"], "y_true": y_slice})
    for model_name, pred in pred_dict.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("pred_dict keys must be non-empty strings.")
        pred_arr = np.asarray(pred)
        if pred_arr.ndim != 1:
            raise ValueError(f"pred_dict['{model_name}'] must be a 1D array.")
        if len(pred_arr) < y_end:
            raise ValueError(
                f"Prediction array for '{model_name}' shorter than y_true slice: "
                f"{len(pred_arr)} < {y_end}."
            )
        pred_slice = pred_arr[y_start:y_end]
        if len(pred_slice) != len(alpha_df):
            raise ValueError(
                f"Prediction length for '{model_name}' does not match y_true slice: "
                f"{len(pred_slice)} vs {len(alpha_df)}."
            )
        alpha_df[f"alpha_{model_name}"] = pred_slice

    md_parquet_path = os.path.join(md_folder, day, f"{skey}.parquet")
    levels = range(1, 6)
    md_columns = ["obe_seq_num"] + [
        f"{side}{level}_{field}"
        for side in ("ask", "bid")
        for level in levels
        for field in ("opx", "qty")
    ]
    md_df = pd.read_parquet(md_parquet_path, columns=md_columns)
    missing_cols = [col for col in md_columns if col not in md_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required MD columns: {missing_cols}.")

    merged = md_df.merge(alpha_df, on="obe_seq_num", how="inner", sort=False)
    merged = merged.sort_values("obe_seq_num").reset_index(drop=True)
    return merged


def mark_strong_weak_ticks(
    df: pd.DataFrame,
    alpha_col: str,
    q: float = 0.95,
) -> pd.DataFrame:
    """Mark strong/weak ticks based on empirical alpha quantiles.

    For the specified alpha column, strong buy ticks are above the q-th
    quantile, strong sell ticks are below the (1 - q)-th quantile, and weak
    ticks are all remaining rows. Quantiles are computed using only the
    provided DataFrame (single instrument / day).
    """
    if alpha_col not in df.columns:
        raise ValueError(f"alpha_col '{alpha_col}' not found in DataFrame.")
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1 (exclusive).")

    q_high = df[alpha_col].quantile(q)
    q_low = df[alpha_col].quantile(1 - q)

    out = df.copy()
    strong_buy_col = f"{alpha_col}_strong_buy"
    strong_sell_col = f"{alpha_col}_strong_sell"
    weak_col = f"{alpha_col}_weak"

    out[strong_buy_col] = out[alpha_col] > q_high
    out[strong_sell_col] = out[alpha_col] < q_low
    out[weak_col] = ~(out[strong_buy_col] | out[strong_sell_col])
    return out


def mark_all_alpha_ticks(df: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """Apply strong/weak tick marking to all alpha_ columns.

    This computes per-column empirical quantiles without cross-model mixing
    and returns a new DataFrame with the added boolean indicator columns.
    """
    alpha_cols = [col for col in df.columns if col.startswith("alpha_")]
    out = df.copy()
    for alpha_col in alpha_cols:
        out = mark_strong_weak_ticks(out, alpha_col, q=q)
    return out
