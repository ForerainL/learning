"""Project-level training entry for adjMidRet60s prediction.

This script adapts to the existing dataset pipeline:
- DayReader performs IO + normalization once per (day, skey)
- DayCache reuses normalized tensors
- TimeSeriesDataset only slices tensors

Evaluation follows the Part 3 requirement exactly:
metrics are computed per (date, stock) and then averaged.
"""

from __future__ import annotations

import copy
import glob
import math
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.cache import DayCache
from dataset.dataset import TimeSeriesDataset
from dataset.index import IndexEntry, build_index
from dataset.reader import DayReader
from model import GRUModel, LSTMModel, TCNModel, TransformerModel

Key = Tuple[str, str]


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class DataConfig:
    data_root: str = "./data"
    stats_path: str = "feature_stats.pt"
    window: int = 64
    cache_size: int = 128
    num_workers: int = 0

    train_start: str = "2022-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-03-31"
    test_start: str = "2024-04-01"
    test_end: str = "2024-12-31"


@dataclass
class OptimConfig:
    name: str = "adamw"  # adam | adamw
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class SchedConfig:
    name: str = "cosine"  # none | cosine | step
    t_max: int = 1000
    step_size: int = 500
    gamma: float = 0.5


@dataclass
class TrainConfig:
    model_name: str = "gru"  # gru | lstm | tcn | transformer
    input_dim: int = 137
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.1

    batch_size: int = 256
    total_steps: int = 5_000
    max_epochs: int = 50
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def iter_days(start: str, end: str) -> Iterable[str]:
    """Yield day strings in [start, end] that also exist on disk."""
    start_d = _to_date(start)
    end_d = _to_date(end)
    cur = start_d
    while cur <= end_d:
        yield cur.isoformat()
        cur += timedelta(days=1)


def discover_skeys(data_root: str, day: str) -> List[str]:
    pattern = os.path.join(data_root, day, "*.parquet")
    files = glob.glob(pattern)
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in files)


def build_meta(data_root: str, days: Sequence[str]) -> Dict[Key, int]:
    """Scan parquet files to obtain sequence lengths T per (day, skey)."""
    meta: Dict[Key, int] = {}
    for day in days:
        for skey in discover_skeys(data_root, day):
            path = os.path.join(data_root, day, f"{skey}.parquet")
            # Only read the label column to keep scanning lightweight.
            frame = pd.read_parquet(path, columns=["label"])
            meta[(day, skey)] = len(frame)
    return meta


def build_days_in_range(cfg: DataConfig, start: str, end: str) -> List[str]:
    """Return only days that actually exist under data_root."""
    days = []
    for day in iter_days(start, end):
        if os.path.isdir(os.path.join(cfg.data_root, day)):
            days.append(day)
    return days


def build_datasets(cfg: TrainConfig) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Construct train/val/test datasets using the existing pipeline."""
    data_cfg = cfg.data
    reader = DayReader(data_cfg.data_root, stats_path=data_cfg.stats_path)
    cache = DayCache(reader, max_size=data_cfg.cache_size)

    train_days = build_days_in_range(data_cfg, data_cfg.train_start, data_cfg.train_end)
    val_days = build_days_in_range(data_cfg, data_cfg.val_start, data_cfg.val_end)
    test_days = build_days_in_range(data_cfg, data_cfg.test_start, data_cfg.test_end)

    train_meta = build_meta(data_cfg.data_root, train_days)
    val_meta = build_meta(data_cfg.data_root, val_days)
    test_meta = build_meta(data_cfg.data_root, test_days)

    train_index = build_index(train_meta, data_cfg.window)
    val_index = build_index(val_meta, data_cfg.window)
    test_index = build_index(test_meta, data_cfg.window)

    train_ds = TimeSeriesDataset(train_index, cache, data_cfg.window)
    val_ds = TimeSeriesDataset(val_index, cache, data_cfg.window)
    test_ds = TimeSeriesDataset(test_index, cache, data_cfg.window)
    return train_ds, val_ds, test_ds


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(cfg)
    data_cfg = cfg.data

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    # Validation/test must not shuffle so we can align indices -> (date, stock).
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def build_model(cfg: TrainConfig) -> nn.Module:
    """Instantiate a model from model.py without normalization layers."""
    from model import ModelConfig

    model_cfg = ModelConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    name = cfg.model_name.lower()
    if name == "gru":
        return GRUModel(model_cfg)
    if name == "lstm":
        return LSTMModel(model_cfg)
    if name == "tcn":
        return TCNModel(model_cfg)
    if name == "transformer":
        return TransformerModel(model_cfg)
    raise ValueError(f"Unknown model_name: {cfg.model_name}")


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.optim.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optim.name}")


def build_scheduler(cfg: TrainConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = cfg.sched.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.sched.t_max)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.sched.step_size, gamma=cfg.sched.gamma)
    raise ValueError(f"Unknown scheduler: {cfg.sched.name}")


# -----------------------------
# Metrics (Part 3 requirement)
# -----------------------------


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    return float(a.corr(b))


def compute_metrics(pred: np.ndarray, label: np.ndarray, days: List[str], skeys: List[str]) -> Dict[str, float]:
    """Compute IC/RankIC/Quantile return per (date, stock) and average."""
    frame = pd.DataFrame(
        {"pred": pred, "label": label, "day": days, "skey": skeys}
    )

    metrics = []
    for (_, _), g in frame.groupby(["day", "skey"], sort=False):
        if len(g) < 5:
            # Very short series leads to unstable rank/correlation stats.
            continue
        ic = _safe_corr(g["pred"], g["label"])
        rank_ic = _safe_corr(g["pred"].rank(), g["label"].rank())

        g_sorted = g.sort_values("pred")
        q = max(int(math.floor(0.1 * len(g_sorted))), 1)
        bottom = g_sorted["label"].iloc[:q].mean()
        top = g_sorted["label"].iloc[-q:].mean()
        qret = float(top - bottom)

        metrics.append({"ic": ic, "rank_ic": rank_ic, "qret": qret})

    if not metrics:
        return {"ic": float("nan"), "rank_ic": float("nan"), "qret": float("nan")}

    m = pd.DataFrame(metrics)
    return {
        "ic": float(m["ic"].mean()),
        "rank_ic": float(m["rank_ic"].mean()),
        "qret": float(m["qret"].mean()),
    }


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Run evaluation and compute group metrics using dataset indices."""
    model.eval()

    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    days: List[str] = []
    skeys: List[str] = []

    # We rely on shuffle=False and the dataset index order.
    index_list: List[IndexEntry] = loader.dataset.index  # type: ignore[attr-defined]
    offset = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out = model(xb).squeeze(-1)
        preds.append(out.detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())

        batch_size = len(yb)
        batch_index = index_list[offset : offset + batch_size]
        days.extend(day for day, _, _ in batch_index)
        skeys.extend(skey for _, skey, _ in batch_index)
        offset += batch_size

    pred_arr = np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)
    label_arr = np.concatenate(labels) if labels else np.empty(0, dtype=np.float32)
    return compute_metrics(pred_arr, label_arr, days, skeys)


# -----------------------------
# Training
# -----------------------------


def train_one_run(cfg: TrainConfig) -> Dict[str, float]:
    """Train a single configuration and return best test metrics."""
    set_seed(cfg.seed)
    device = cfg.device

    train_loader, val_loader, test_loader = build_loaders(cfg)

    model = build_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    best_state = copy.deepcopy(model.state_dict())
    best_val_ic = -float("inf")
    steps = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            steps += 1
            if steps >= cfg.total_steps:
                break
        val_metrics = evaluate_epoch(model, val_loader, device)
        val_ic = val_metrics["ic"]

        if not math.isnan(val_ic) and val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = copy.deepcopy(model.state_dict())

        if steps >= cfg.total_steps:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_epoch(model, test_loader, device)
    # Include the monitoring metric for logging/comparison.
    test_metrics["best_val_ic"] = float(best_val_ic)
    return test_metrics


# -----------------------------
# Hyperparameter search
# -----------------------------


def hyperparameter_search(base_cfg: TrainConfig, search_space: Dict[str, Sequence], max_runs: int | None = None) -> List[Dict[str, float]]:
    """Simple grid search over provided parameter lists."""
    keys = list(search_space.keys())
    grids = [list(search_space[k]) for k in keys]
    total = int(np.prod([len(g) for g in grids])) if grids else 1

    results: List[Dict[str, float]] = []
    run_count = 0

    def _assign(cfg: TrainConfig, key: str, value) -> TrainConfig:
        if key.startswith("optim."):
            return replace(cfg, optim=replace(cfg.optim, **{key.split(".", 1)[1]: value}))
        if key.startswith("sched."):
            return replace(cfg, sched=replace(cfg.sched, **{key.split(".", 1)[1]: value}))
        if key.startswith("data."):
            return replace(cfg, data=replace(cfg.data, **{key.split(".", 1)[1]: value}))
        return replace(cfg, **{key: value})

    for flat_idx in range(total):
        if max_runs is not None and run_count >= max_runs:
            break
        cfg = base_cfg
        idx = flat_idx
        for key, grid in zip(keys, grids):
            if not grid:
                continue
            choice = grid[idx % len(grid)]
            idx //= len(grid)
            cfg = _assign(cfg, key, choice)

        metrics = train_one_run(cfg)
        result = {
            "model": cfg.model_name,
            "lr": cfg.optim.lr,
            "batch_size": cfg.batch_size,
            "total_steps": cfg.total_steps,
            "weight_decay": cfg.optim.weight_decay,
            "dropout": cfg.dropout,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            **metrics,
        }
        results.append(result)
        run_count += 1

    return results


# -----------------------------
# Main entry
# -----------------------------


def main() -> None:
    cfg = TrainConfig()

    # Example search space; adjust as needed for real experiments.
    search_space = {
        "model_name": ["gru", "lstm", "tcn", "transformer"],
        "optim.lr": [5e-4, 1e-3],
        "batch_size": [256],
        "total_steps": [3_000],
        "optim.weight_decay": [1e-4],
        "dropout": [0.05, 0.1],
        "hidden_dim": [16, 24],
        "num_layers": [1, 2],
    }

    results = hyperparameter_search(cfg, search_space, max_runs=4)
    results_df = pd.DataFrame(results).sort_values("ic", ascending=False)
    print(results_df.head(10))


if __name__ == "__main__":
    main()
