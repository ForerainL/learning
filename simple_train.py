"""Standalone, minimal training loop for quick testing.

This file intentionally does NOT import anything from train.py.
It adapts to the existing dataset pipeline:
- DayReader: IO + normalization
- DayCache: caching per (day, skey)
- TimeSeriesDataset: tensor slicing only
"""

from __future__ import annotations

import copy
import glob
import math
import os
from dataclasses import dataclass, field
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
from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel

Key = Tuple[str, str]


# ======================================================
# Config
# ======================================================


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


@dataclass
class SimpleTrainConfig:
    # model
    model_name: str = "gru"  # gru | lstm | tcn | transformer
    input_dim: int = 137
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.1

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256

    # training control
    max_epochs: int = 20
    patience: int = 5
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # outputs
    save_dir: str = "./artifacts"
    run_name: str = "simple_run"

    data: DataConfig = field(default_factory=DataConfig)


# ======================================================
# Data helpers (standalone)
# ======================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def iter_days(start: str, end: str) -> Iterable[str]:
    cur = _to_date(start)
    end_d = _to_date(end)
    while cur <= end_d:
        yield cur.isoformat()
        cur += timedelta(days=1)


def discover_skeys(data_root: str, day: str) -> List[str]:
    pattern = os.path.join(data_root, day, "*.parquet")
    files = glob.glob(pattern)
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in files)


def build_days_in_range(cfg: DataConfig, start: str, end: str) -> List[str]:
    days: List[str] = []
    for day in iter_days(start, end):
        if os.path.isdir(os.path.join(cfg.data_root, day)):
            days.append(day)
    return days


def build_meta(data_root: str, days: Sequence[str]) -> Dict[Key, int]:
    """Scan only the label column to get sequence lengths."""
    meta: Dict[Key, int] = {}
    for day in days:
        for skey in discover_skeys(data_root, day):
            path = os.path.join(data_root, day, f"{skey}.parquet")
            frame = pd.read_parquet(path, columns=["label"])
            meta[(day, skey)] = len(frame)
    return meta


def build_loaders(cfg: SimpleTrainConfig) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg.data
    reader = DayReader(data_cfg.data_root, stats_path=data_cfg.stats_path)
    cache = DayCache(reader, max_size=data_cfg.cache_size)

    train_days = build_days_in_range(data_cfg, data_cfg.train_start, data_cfg.train_end)
    val_days = build_days_in_range(data_cfg, data_cfg.val_start, data_cfg.val_end)

    train_meta = build_meta(data_cfg.data_root, train_days)
    val_meta = build_meta(data_cfg.data_root, val_days)

    train_index = build_index(train_meta, data_cfg.window)
    val_index = build_index(val_meta, data_cfg.window)

    train_ds = TimeSeriesDataset(train_index, cache, data_cfg.window)
    val_ds = TimeSeriesDataset(val_index, cache, data_cfg.window)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    # Keep val_loader ordered so we can map batches to dataset.index.
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


# ======================================================
# Model / optim helpers (standalone)
# ======================================================


def build_model(cfg: SimpleTrainConfig) -> nn.Module:
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


def build_optimizer(cfg: SimpleTrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ======================================================
# Evaluation utils
# ======================================================


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(values))


def compute_group_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Part 3 metrics per (date, stock), then average across groups."""
    ic_list: List[float] = []
    rankic_list: List[float] = []
    qret_list: List[float] = []

    for (_, _), g in df.groupby(["date", "stock"], sort=False):
        if len(g) < 5:
            continue
        ic = float(g["pred"].corr(g["y"]))
        rankic = float(g["pred"].rank().corr(g["y"].rank()))

        g_sorted = g.sort_values("pred")
        n = len(g_sorted)
        q = max(int(0.1 * n), 1)
        qret = float(g_sorted["y"].iloc[-q:].mean() - g_sorted["y"].iloc[:q].mean())

        ic_list.append(ic)
        rankic_list.append(rankic)
        qret_list.append(qret)

    return {
        "IC": _safe_mean(ic_list),
        "RankIC": _safe_mean(rankic_list),
        "QuantileReturn": _safe_mean(qret_list),
    }


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str) -> pd.DataFrame:
    """Collect predictions with (date, stock) from dataset.index order."""
    model.eval()

    dataset: TimeSeriesDataset = loader.dataset  # type: ignore[assignment]
    index_list: List[IndexEntry] = dataset.index  # type: ignore[attr-defined]

    rows: List[Dict[str, float | str]] = []
    offset = 0

    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).squeeze(-1).detach().cpu().numpy()
        y = yb.detach().cpu().numpy()

        batch_size = len(y)
        batch_index = index_list[offset : offset + batch_size]
        offset += batch_size

        for i, (day, skey, _) in enumerate(batch_index):
            rows.append(
                {
                    "date": day,
                    "stock": skey,
                    "y": float(y[i]),
                    "pred": float(pred[i]),
                }
            )

    return pd.DataFrame(rows)


def compute_loss(model: nn.Module, loader: DataLoader, device: str, loss_fn: nn.Module) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


# ======================================================
# Training
# ======================================================


def _is_improved(metric: str, score: float, best: float) -> bool:
    if math.isnan(score):
        return False
    if metric == "val_loss":
        return score < best
    return score > best


def train(cfg: SimpleTrainConfig) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, Dict[str, float]]:
    """Train with logging + early stopping and save artifacts."""
    set_seed(cfg.seed)
    device = cfg.device

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    loss_fn = nn.MSELoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_IC": [],
        "val_RankIC": [],
        "val_QuantileReturn": [],
    }

    best_score = float("inf") if cfg.early_stop_metric == "val_loss" else -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}
    patience_cnt = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss = compute_loss(model, val_loader, device, loss_fn)
        val_df = collect_predictions(model, val_loader, device)
        val_metrics = compute_group_metrics(val_df)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_IC"].append(val_metrics["IC"])
        history["val_RankIC"].append(val_metrics["RankIC"])
        history["val_QuantileReturn"].append(val_metrics["QuantileReturn"])

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"IC={val_metrics['IC']:.4f}"
        )

        score = val_loss if cfg.early_stop_metric == "val_loss" else val_metrics["IC"]
        if _is_improved(cfg.early_stop_metric, score, best_score):
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history_df = pd.DataFrame(history)

    os.makedirs(cfg.save_dir, exist_ok=True)
    model_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
    hist_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_history.csv")
    metrics_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_val_metrics.json")

    torch.save(best_state, model_path)
    history_df.to_csv(hist_path, index=False)
    pd.Series(best_val_metrics).to_json(metrics_path, force_ascii=False, indent=2)

    print(f"Saved best model to: {model_path}")
    print(f"Saved history to:    {hist_path}")
    print(f"Saved metrics to:    {metrics_path}")

    return best_state, history_df, best_val_metrics


if __name__ == "__main__":
    cfg = SimpleTrainConfig()
    train(cfg)
