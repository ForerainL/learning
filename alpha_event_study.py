"""Utilities for alpha-driven event study analysis on market microstructure data."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd


def classify_alpha_events(
    alpha: pd.Series,
    q_high: float = 0.99,
    q_low: float = 0.01,
    q_mid: float = 0.50,
) -> pd.Series:
    """Classify alpha values into event categories based on quantiles."""
    if not isinstance(alpha, pd.Series):
        raise ValueError("alpha must be a pandas Series.")
    if not 0 < q_low < q_mid < q_high < 1:
        raise ValueError("Quantiles must satisfy 0 < q_low < q_mid < q_high < 1.")

    valid_alpha = alpha.dropna()
    if valid_alpha.empty:
        return pd.Series(index=alpha.index, dtype="object")

    high = float(valid_alpha.quantile(q_high))
    low = float(valid_alpha.quantile(q_low))
    mid = float(valid_alpha.abs().quantile(q_mid))

    labels = pd.Series("other", index=alpha.index, dtype="object")
    labels.loc[alpha >= high] = "strong_buy"
    labels.loc[alpha <= low] = "strong_sell"
    labels.loc[alpha.abs() <= mid] = "neutral"
    return labels


def align_alpha_with_md(
    md_df: pd.DataFrame,
    alpha: pd.Series,
    on: str = "obe_seq_num",
) -> pd.DataFrame:
    """Left-join alpha onto md_df and drop rows with missing alpha."""
    if not isinstance(md_df, pd.DataFrame):
        raise ValueError("md_df must be a DataFrame.")
    if not isinstance(alpha, pd.Series):
        raise ValueError("alpha must be a Series.")
    if on not in md_df.columns:
        raise ValueError(f"md_df must contain join column '{on}'.")

    alpha_df = alpha.rename("alpha").to_frame()
    alpha_df[on] = alpha_df.index
    merged = md_df.merge(alpha_df, on=on, how="left", sort=False)
    merged = merged.dropna(subset=["alpha"]).reset_index(drop=True)
    return merged


def extract_event_windows(
    df: pd.DataFrame,
    event_mask: pd.Series,
    window: int = 20,
    seq_col: str = "obe_seq_num",
) -> Dict[int, pd.DataFrame]:
    """Extract symmetric windows around each event using sequence order."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame.")
    if not isinstance(event_mask, pd.Series):
        raise ValueError("event_mask must be a Series.")
    if window <= 0:
        raise ValueError("window must be positive.")
    if seq_col not in df.columns:
        raise ValueError(f"df must contain sequence column '{seq_col}'.")

    if not event_mask.index.equals(df.index):
        event_mask = event_mask.reindex(df.index, fill_value=False)

    df_sorted = df.sort_values(seq_col).reset_index(drop=True)
    event_mask_sorted = event_mask.loc[df_sorted.index]
    event_indices = np.flatnonzero(event_mask_sorted.to_numpy())

    windows: Dict[int, pd.DataFrame] = {}
    n = len(df_sorted)
    for idx in event_indices:
        start = idx - window
        end = idx + window
        if start < 0 or end >= n:
            continue
        window_df = df_sorted.iloc[start : end + 1].copy()
        event_seq = int(window_df[seq_col].iloc[window])
        windows[event_seq] = window_df
    return windows


def compute_mid_drift(window_df: pd.DataFrame, mid_col: str) -> dict:
    """Summarize pre/post mid-price drift around the event."""
    mid = window_df[mid_col].to_numpy()
    if len(mid) < 3:
        return {"mid_pre": np.nan, "mid_post": np.nan, "mid_drift": np.nan}
    mid_pre = float(mid[0])
    mid_post = float(mid[-1])
    return {"mid_pre": mid_pre, "mid_post": mid_post, "mid_drift": mid_post - mid_pre}


def compute_spread_dynamics(window_df: pd.DataFrame, spread_col: str) -> dict:
    """Detect spread widening or narrowing around the event."""
    spread = window_df[spread_col].to_numpy()
    if len(spread) < 3:
        return {"spread_pre": np.nan, "spread_post": np.nan, "spread_change": np.nan}
    spread_pre = float(spread[0])
    spread_post = float(spread[-1])
    return {
        "spread_pre": spread_pre,
        "spread_post": spread_post,
        "spread_change": spread_post - spread_pre,
    }


def compute_depth_imbalance_stats(
    window_df: pd.DataFrame,
    bid_depth_col: str,
    ask_depth_col: str,
) -> dict:
    """Summarize depth imbalance buildup before the event."""
    bid = window_df[bid_depth_col].to_numpy()
    ask = window_df[ask_depth_col].to_numpy()
    denom = bid + ask
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(denom == 0, np.nan, (bid - ask) / denom)
    if imbalance.size == 0:
        return {"imbalance_mean": np.nan, "imbalance_pre": np.nan, "imbalance_post": np.nan}
    return {
        "imbalance_mean": float(np.nanmean(imbalance)),
        "imbalance_pre": float(imbalance[0]),
        "imbalance_post": float(imbalance[-1]),
    }


def compute_trade_direction_stats(
    window_df: pd.DataFrame,
    trade_dir_col: str,
) -> dict:
    """Check whether trades are directionally aggressive around the event."""
    trade_dir = window_df[trade_dir_col].to_numpy()
    if trade_dir.size == 0:
        return {"trade_dir_mean": np.nan, "trade_dir_pre": np.nan, "trade_dir_post": np.nan}
    return {
        "trade_dir_mean": float(np.nanmean(trade_dir)),
        "trade_dir_pre": float(trade_dir[0]),
        "trade_dir_post": float(trade_dir[-1]),
    }


def compute_cancel_activity(
    window_df: pd.DataFrame,
    cancel_col: str,
) -> dict:
    """Detect abnormal cancel activity around the event."""
    cancels = window_df[cancel_col].to_numpy()
    if cancels.size == 0:
        return {"cancel_mean": np.nan, "cancel_pre": np.nan, "cancel_post": np.nan}
    return {
        "cancel_mean": float(np.nanmean(cancels)),
        "cancel_pre": float(cancels[0]),
        "cancel_post": float(cancels[-1]),
    }


def aggregate_event_statistics(
    event_windows: Dict[int, pd.DataFrame],
    stat_fns: Iterable[Callable[[pd.DataFrame], dict]],
) -> pd.DataFrame:
    """Apply statistic functions to each event window and aggregate results."""
    rows = []
    for event_seq, window_df in event_windows.items():
        stats: Dict[str, float] = {"event_seq_num": event_seq}
        for fn in stat_fns:
            stats.update(fn(window_df))
        rows.append(stats)
    return pd.DataFrame(rows)


def plot_average_event_path(
    event_windows: Dict[int, pd.DataFrame],
    value_col: str,
    title: str = "",
):
    """Plot the average event-window path for a given value column."""
    import matplotlib.pyplot as plt

    if not event_windows:
        raise ValueError("event_windows must be non-empty.")

    paths = [window_df[value_col].to_numpy() for window_df in event_windows.values()]
    lengths = {len(path) for path in paths}
    if len(lengths) != 1:
        raise ValueError("All windows must have the same length for plotting.")

    data = np.vstack(paths)
    avg = np.nanmean(data, axis=0)
    x = np.arange(len(avg)) - (len(avg) // 2)

    plt.plot(x, avg, linewidth=1.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event window index")
    plt.ylabel(value_col)
    return plt.gca()
