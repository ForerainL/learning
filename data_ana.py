"""Analysis utilities for aligning predictions with market data."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def build_md_alpha_df(
    x_parquet_path: str,
    md_parquet_path: str,
    y_all: np.ndarray,
    pred_dict: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build a merged DataFrame of order book data and model predictions.

    The function aligns predictions with market data strictly by `obe_seq_num`:
    - Reads x.parquet, drops rows with missing y, and assumes y_all/pred_y
      are already aligned with the filtered rows.
    - Builds an alpha table with obe_seq_num, y_true, and alpha_<model> columns.
    - Reads market data parquet, keeping only the top 5 levels of bid/ask data.
    - Merges by obe_seq_num (inner join) without filling or padding gaps.

    Args:
        x_parquet_path: Path to the x.parquet file (must include obe_seq_num and y).
        md_parquet_path: Path to the market data parquet file.
        y_all: 1D numpy array of true labels aligned with filtered x.parquet rows.
        pred_dict: Mapping of model name to 1D prediction arrays aligned with y_all.

    Returns:
        DataFrame sorted by obe_seq_num containing order book levels, y_true,
        and alpha_<model_name> columns.
    """
    if not isinstance(x_parquet_path, str) or not x_parquet_path:
        raise ValueError("x_parquet_path must be a non-empty string.")
    if not isinstance(md_parquet_path, str) or not md_parquet_path:
        raise ValueError("md_parquet_path must be a non-empty string.")

    y_all_arr = np.asarray(y_all)
    if y_all_arr.ndim != 1:
        raise ValueError("y_all must be a 1D array.")

    x_df = pd.read_parquet(x_parquet_path, columns=["obe_seq_num", "y"])
    x_df = x_df.dropna(subset=["y"]).reset_index(drop=True)

    if len(x_df) != len(y_all_arr):
        raise ValueError(
            "y_all length does not match filtered x.parquet rows: "
            f"{len(y_all_arr)} vs {len(x_df)}."
        )

    alpha_df = pd.DataFrame({"obe_seq_num": x_df["obe_seq_num"], "y_true": y_all_arr})

    for model_name, pred in pred_dict.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("pred_dict keys must be non-empty strings.")
        pred_arr = np.asarray(pred)
        if pred_arr.ndim != 1:
            raise ValueError(f"pred_dict['{model_name}'] must be a 1D array.")
        if len(pred_arr) != len(alpha_df):
            raise ValueError(
                f"Prediction length for '{model_name}' does not match y_all: "
                f"{len(pred_arr)} vs {len(alpha_df)}."
            )
        alpha_df[f"alpha_{model_name}"] = pred_arr

    levels = range(1, 6)
    md_columns = ["obe_seq_num"] + [
        f"{side}_{field}_{level}"
        for side in ("bid", "ask")
        for field in ("price", "size")
        for level in levels
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
