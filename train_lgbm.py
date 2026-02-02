"""Train and evaluate a LightGBM baseline on global tensors."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import torch


@dataclass
class LGBMConfig:
    tensor_root: str = "./global_data"
    window: int = 64
    max_epochs: int = 500
    patience: int = 20
    val_every: int = 1
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    save_dir: str = "./artifacts"

    train_start: str = "20220101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240331"
    test_start: str = "20240401"
    test_end: str = "20241231"

    num_threads: int = 64
    num_leaves: int = 63
    max_depth: int = 8
    min_data_in_leaf: int = 20000
    max_bin: int = 63
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    learning_rate: float = 0.05
    verbosity: int = -1

    x_all_name: str = "X_all.pt"
    y_all_name: str = "y_all.pt"
    x_test_name: str = "X_test.pt"
    y_test_name: str = "y_test.pt"
    segment_name: str = "segment.pkl"
    segment_test_name: str = "segment_test.pkl"


def load_tensor(path: str) -> np.ndarray:
    """Load a torch tensor and cast to float32 numpy."""
    tensor = torch.load(path, map_location="cpu")
    return tensor.detach().cpu().numpy().astype(np.float32)


def load_segments(path: str) -> List[Dict[str, object]]:
    """Load segmentation metadata for per-day ranges."""
    with open(path, "rb") as f:
        return pickle.load(f)


def indices_for_day_range(
    segments: Sequence[Dict[str, object]],
    start: str,
    end: str,
) -> np.ndarray:
    """Return indices covering all rows for days in the inclusive range."""
    indices: List[int] = []
    for seg in segments:
        day = str(seg["day"])
        if not (start <= day <= end):
            continue
        seg_start = int(seg["start_idx"])
        length = int(seg["length"])
        indices.extend(range(seg_start, seg_start + length))
    return np.asarray(indices, dtype=np.int64)


def masked_indices_for_day_range(
    segments: Sequence[Dict[str, object]],
    start: str,
    end: str,
    window: int,
) -> np.ndarray:
    """Return indices excluding the first `window` ticks of each day."""
    indices: List[int] = []
    for seg in segments:
        day = str(seg["day"])
        if not (start <= day <= end):
            continue
        seg_start = int(seg["start_idx"])
        length = int(seg["length"])
        start_idx = seg_start + window
        end_idx = seg_start + length
        if start_idx >= end_idx:
            continue
        indices.extend(range(start_idx, end_idx))
    return np.asarray(indices, dtype=np.int64)


def compute_metrics(pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute MSE and IC on aligned arrays."""
    if len(pred) == 0:
        return {"MSE": float("nan"), "IC": float("nan")}
    mse = float(np.mean((pred - y_true) ** 2))
    if len(pred) < 2:
        ic = float("nan")
    else:
        ic = float(np.corrcoef(pred, y_true)[0, 1])
    return {"MSE": mse, "IC": ic}


def build_run_name(cfg: LGBMConfig) -> str:
    """Generate a readable run name with key hyperparameters."""
    return (
        "lgbm_"
        f"lr{cfg.learning_rate}_"
        f"leaves{cfg.num_leaves}_"
        f"minleaf{cfg.min_data_in_leaf}_"
        f"depth{cfg.max_depth}"
    )


def train_lgbm(cfg: LGBMConfig) -> None:
    """Train LightGBM on full rows, evaluate with per-day masking."""
    os.makedirs(cfg.save_dir, exist_ok=True)

    x_all_path = os.path.join(cfg.tensor_root, cfg.x_all_name)
    y_all_path = os.path.join(cfg.tensor_root, cfg.y_all_name)
    x_test_path = os.path.join(cfg.tensor_root, cfg.x_test_name)
    y_test_path = os.path.join(cfg.tensor_root, cfg.y_test_name)
    segment_path = os.path.join(cfg.tensor_root, cfg.segment_name)
    segment_test_path = os.path.join(cfg.tensor_root, cfg.segment_test_name)

    x_all = load_tensor(x_all_path)
    y_all = load_tensor(y_all_path).reshape(-1)
    x_test = load_tensor(x_test_path)
    y_test = load_tensor(y_test_path).reshape(-1)

    segments = load_segments(segment_path)
    segments_test = load_segments(segment_test_path)

    train_idx = indices_for_day_range(segments, cfg.train_start, cfg.train_end)
    val_fit_idx = indices_for_day_range(segments, cfg.val_start, cfg.val_end)
    val_eval_idx = masked_indices_for_day_range(segments, cfg.val_start, cfg.val_end, cfg.window)
    test_eval_idx = masked_indices_for_day_range(segments_test, cfg.test_start, cfg.test_end, cfg.window)

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val_fit = x_all[val_fit_idx]
    y_val_fit = y_all[val_fit_idx]
    x_val_eval = x_all[val_eval_idx]
    y_val_eval = y_all[val_eval_idx]
    x_test_eval = x_test[test_eval_idx]
    y_test_eval = y_test[test_eval_idx]

    train_dataset = lgb.Dataset(x_train, label=y_train)
    val_dataset = lgb.Dataset(x_val_fit, label=y_val_fit, reference=train_dataset)

    run_name = build_run_name(cfg)
    train_bin_path = os.path.join(cfg.save_dir, f"{run_name}_train.bin")
    val_bin_path = os.path.join(cfg.save_dir, f"{run_name}_val.bin")
    train_dataset.save_binary(train_bin_path)
    val_dataset.save_binary(val_bin_path)

    params = dict(
        boosting_type="gbdt",
        objective="regression",
        metric="None",
        tree_learner="data",
        num_threads=cfg.num_threads,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_data_in_leaf=cfg.min_data_in_leaf,
        max_bin=cfg.max_bin,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        learning_rate=cfg.learning_rate,
        verbosity=cfg.verbosity,
    )

    def feval(preds: np.ndarray, dataset: lgb.Dataset) -> List[Tuple[str, float, bool]]:
        labels = dataset.get_label()
        metrics = compute_metrics(preds, labels)
        return [
            ("MSE", metrics["MSE"], False),
            ("IC", metrics["IC"], True),
        ]

    booster = lgb.train(
        params,
        train_set=train_dataset,
        num_boost_round=cfg.max_epochs,
        valid_sets=[val_dataset],
        valid_names=["valid"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(cfg.patience),
            lgb.log_evaluation(cfg.val_every),
        ],
    )

    best_iteration = booster.best_iteration
    val_pred = booster.predict(x_val_eval, num_iteration=best_iteration)
    val_metrics = compute_metrics(val_pred, y_val_eval)

    test_pred = booster.predict(x_test_eval, num_iteration=best_iteration)
    test_metrics = compute_metrics(test_pred, y_test_eval)

    model_path = os.path.join(cfg.save_dir, f"{run_name}_model.txt")
    val_metrics_path = os.path.join(cfg.save_dir, f"{run_name}_val_metrics.json")
    test_metrics_path = os.path.join(cfg.save_dir, f"{run_name}_test_metrics.json")
    test_pred_path = os.path.join(cfg.save_dir, f"{run_name}_test_pred.npy")

    booster.save_model(model_path, num_iteration=best_iteration)
    np.save(test_pred_path, test_pred)

    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved train dataset: {train_bin_path}")
    print(f"Saved val dataset: {val_bin_path}")
    print(f"Saved val metrics: {val_metrics_path}")
    print(f"Saved test metrics: {test_metrics_path}")
    print(f"Saved test predictions: {test_pred_path}")


if __name__ == "__main__":
    train_lgbm(LGBMConfig())
