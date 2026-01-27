"""Exploratory analysis for Part 1 using training data only.

This script:
1) Loads all training parquet files into a single DataFrame.
2) Drops rows with missing adjMidRet60s labels.
3) Classifies features into heavy-tailed, smooth, or discrete groups.
4) Applies leakage-safe normalization based only on training stats.
5) Plots ACFs for the label and representative features.
6) Runs an OLS regression for interpretability.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from statsmodels.tsa.stattools import acf

TRAIN_START = datetime(2022, 1, 1)
TRAIN_END = datetime(2023, 12, 31)
TARGET_LABEL = "adjMidRet60s"
FEATURE_PREFIX = "x_"
MAX_LAGS = 50
RANDOM_SEED = 42
STATS_PATH = "feature_stats.pt"


@dataclass
class FeatureGroups:
    heavy_tailed: List[str]
    smooth: List[str]
    discrete: List[str]


def list_training_files(data_root: str) -> List[str]:
    """Collect parquet files in date folders and keep training range only."""
    pattern = os.path.join(data_root, "*", "*.parquet")
    files = []
    for path in glob.glob(pattern):
        date_str = os.path.basename(os.path.dirname(path))
        try:
            date_val = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if TRAIN_START <= date_val <= TRAIN_END:
            files.append(path)
    return sorted(files)


def load_training_data(paths: Iterable[str]) -> pd.DataFrame:
    """Read all training parquet files into a single DataFrame."""
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise ValueError("No training parquet files found for the given date range.")
    return pd.concat(frames, ignore_index=True)


def select_features_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep feature columns and the target label only."""
    feature_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    if TARGET_LABEL not in df.columns:
        raise ValueError(f"Target label '{TARGET_LABEL}' not found in data columns.")
    return df[feature_cols + [TARGET_LABEL]]


def drop_missing_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing labels to avoid training on unknown targets."""
    # In finance, missing labels often reflect unavailable or invalid returns;
    # dropping prevents leaking information from imputed targets.
    return df.dropna(subset=[TARGET_LABEL])


def classify_features(df: pd.DataFrame, feature_cols: List[str]) -> FeatureGroups:
    """Classify features by simple distributional statistics."""
    skewness = df[feature_cols].skew()
    kurtosis = df[feature_cols].kurtosis()
    nunique = df[feature_cols].nunique(dropna=True)
    n_rows = len(df)

    discrete = []
    heavy_tailed = []
    smooth = []

    for col in feature_cols:
        unique_ratio = nunique[col] / max(n_rows, 1)
        if nunique[col] <= 10 or unique_ratio < 0.01:
            discrete.append(col)
        elif abs(skewness[col]) > 1.5 or kurtosis[col] > 6.0:
            heavy_tailed.append(col)
        else:
            smooth.append(col)

    return FeatureGroups(heavy_tailed=heavy_tailed, smooth=smooth, discrete=discrete)


def robust_normalize(series: pd.Series) -> pd.Series:
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return series - median
    return (series - median) / iqr


def zscore_normalize(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series - mean
    return (series - mean) / std


def normalize_features(df: pd.DataFrame, groups: FeatureGroups) -> pd.DataFrame:
    """Normalize by feature type using training-only statistics."""
    normalized = df.copy()

    # Heavy-tailed features benefit from robust scaling to reduce the influence
    # of extreme events often seen in intraday returns and volume spikes.
    for col in groups.heavy_tailed:
        normalized[col] = robust_normalize(normalized[col])

    # Smooth features are roughly Gaussian; z-score preserves relative scale.
    for col in groups.smooth:
        normalized[col] = zscore_normalize(normalized[col])

    # Discrete features (e.g., flags/levels) should remain unscaled to preserve
    # their categorical meaning.
    return normalized


def _feature_range(start: int, end: int) -> List[str]:
    return [f"x_{i:03d}" for i in range(start, end + 1)]


def compute_feature_stats(df: pd.DataFrame, groups: FeatureGroups) -> dict:
    """Compute per-feature stats with fixed type lists and clipping."""
    zscore_cols = (
        _feature_range(10, 31)
        + _feature_range(34, 37)
        + _feature_range(57, 58)
        + ["x_045", "x_050"]
        + _feature_range(63, 70)
    )
    right_skew_cols = (
        _feature_range(32, 33)
        + _feature_range(38, 40)
        + _feature_range(43, 44)
        + _feature_range(51, 56)
        + _feature_range(59, 60)
    )
    all_cols = [f"x_{i:03d}" for i in range(137)]
    scale_cols = [
        col for col in all_cols if col not in zscore_cols and col not in right_skew_cols
    ]

    q_low = 0.01
    q_high = 0.99
    feature_stats: dict = {}

    for col in all_cols:
        series = df[col].astype(float)
        clip_low = float(series.quantile(q_low))
        clip_high = float(series.quantile(q_high))
        clipped = series.clip(lower=clip_low, upper=clip_high)

        if col in zscore_cols:
            mean = float(clipped.mean())
            std = float(clipped.std(ddof=0))
            feature_stats[col] = {
                "type": "zscore",
                "clip_low": clip_low,
                "clip_high": clip_high,
                "mean": mean,
                "std": std if std != 0 else 1.0,
            }
        elif col in right_skew_cols:
            logged = np.log1p(clipped)
            log_mean = float(logged.mean())
            log_std = float(logged.std(ddof=0))
            feature_stats[col] = {
                "type": "right_skew",
                "clip_low": clip_low,
                "clip_high": clip_high,
                "log_mean": log_mean,
                "log_std": log_std if log_std != 0 else 1.0,
            }
        else:
            max_abs = float(np.abs(clipped).max())
            feature_stats[col] = {
                "type": "scale",
                "clip_low": clip_low,
                "clip_high": clip_high,
                "max_abs": max_abs if max_abs != 0 else 1.0,
            }

    return feature_stats


def save_feature_stats(df: pd.DataFrame, groups: FeatureGroups, path: str) -> dict:
    """Persist feature statistics for the training pipeline."""
    stats = compute_feature_stats(df, groups)
    torch.save(stats, path)
    return stats


def plot_acf(series: pd.Series, title: str, ax: plt.Axes) -> None:
    values = series.dropna().values
    if len(values) == 0:
        ax.set_title(f"{title} (no data)")
        return
    acf_vals = acf(values, nlags=MAX_LAGS, fft=True)
    ax.stem(range(len(acf_vals)), acf_vals, use_line_collection=True)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")


def plot_acfs(df: pd.DataFrame, groups: FeatureGroups) -> None:
    """Plot ACF for the label and representative features from each group."""
    rng = np.random.default_rng(RANDOM_SEED)

    reps = []
    for group_name, features in [
        ("Heavy-tailed", groups.heavy_tailed),
        ("Smooth", groups.smooth),
        ("Discrete", groups.discrete),
    ]:
        if features:
            sample = rng.choice(features, size=min(3, len(features)), replace=False)
            reps.extend([(group_name, feat) for feat in sample])

    n_plots = 1 + len(reps)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    plot_acf(df[TARGET_LABEL], f"ACF: {TARGET_LABEL}", axes[0])

    for idx, (group_name, feat) in enumerate(reps, start=1):
        plot_acf(df[feat], f"ACF: {feat} ({group_name})", axes[idx])

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def run_ols(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Fit OLS for interpretability, not for predictive performance."""
    X = df[feature_cols]
    y = df[TARGET_LABEL]
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    print("OLS coefficients:")
    print(results.params)
    print("\nOLS t-statistics:")
    print(results.tvalues)
    print(f"\nR-squared: {results.rsquared:.4f}")


def compute_feature_ic_rankic(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Compute IC (Pearson) and RankIC (Spearman) per feature and sort."""
    label = df[TARGET_LABEL]
    results = []
    for col in feature_cols:
        feature = df[col]
        ic = feature.corr(label, method="pearson")
        rank_ic = feature.corr(label, method="spearman")
        results.append({"feature": col, "ic": ic, "rank_ic": rank_ic})

    result_df = pd.DataFrame(results).sort_values(
        by=["ic", "rank_ic"], ascending=False
    )
    return result_df


def report_feature_ic_rankic(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Compute and report sorted IC/RankIC for individual features."""
    ic_table = compute_feature_ic_rankic(df, feature_cols)
    print("\nFeature IC/RankIC (sorted by IC then RankIC):")
    print(ic_table.head(20))


def main() -> None:
    data_root = os.environ.get("DATA_ROOT", "./data")

    paths = list_training_files(data_root)
    print(f"Found {len(paths)} training parquet files.")

    df_raw = load_training_data(paths)
    df_selected = select_features_and_label(df_raw)

    df_clean = drop_missing_labels(df_selected)

    feature_cols = [col for col in df_clean.columns if col.startswith(FEATURE_PREFIX)]
    groups = classify_features(df_clean, feature_cols)

    print(f"Heavy-tailed features: {len(groups.heavy_tailed)}")
    print(f"Smooth features: {len(groups.smooth)}")
    print(f"Discrete features: {len(groups.discrete)}")

    save_feature_stats(df_clean, groups, STATS_PATH)
    print(f"Saved feature statistics to {STATS_PATH}.")

    df_normalized = normalize_features(df_clean, groups)

    plot_acfs(df_normalized, groups)
    report_feature_ic_rankic(df_normalized, feature_cols)
    run_ols(df_normalized, feature_cols)


if __name__ == "__main__":
    main()
