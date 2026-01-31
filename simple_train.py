"""Standalone, minimal training loop for quick testing.

This file intentionally does NOT import anything from train.py.
It adapts to the existing dataset pipeline:
- DayReader: IO + normalization
- DayCache: caching per (day, skey)
- TimeSeriesDataset: tensor slicing only
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

    train_start: str = "20220101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240331"
    test_start: str = "20240401"
    test_end: str = "20241231"


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
    val_interval: int = 1  # run validation every N epochs
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


def discover_all_days(data_root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(data_root) if d.isdigit() and len(d) == 8
    )


def discover_skeys(data_root: str, day: str) -> List[str]:
    pattern = os.path.join(data_root, day, "*.parquet")
    files = glob.glob(pattern)
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in files)


def build_meta(data_root: str, days: Sequence[str]) -> Dict[Key, int]:
    """Scan only the label column to get sequence lengths."""
    meta: Dict[Key, int] = {}
    for day in days:
        for skey in discover_skeys(data_root, day):
            path = os.path.join(data_root, day, f"{skey}.parquet")
            frame = pd.read_parquet(path, columns=["label"])
            meta[(day, skey)] = len(frame)
    return meta


def _make_loader(
    days: List[str],
    cfg: SimpleTrainConfig,
    cache: DayCache,
    shuffle: bool,
) -> DataLoader:
    data_cfg = cfg.data
    meta = build_meta(data_cfg.data_root, days)
    index = build_index(meta, data_cfg.window)
    ds = TimeSeriesDataset(index, cache, data_cfg.window)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        drop_last=False,
    )


def build_loaders(cfg: SimpleTrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg.data
    reader = DayReader(data_cfg.data_root, stats_path=data_cfg.stats_path)
    cache = DayCache(reader, max_size=data_cfg.cache_size)

    all_days = discover_all_days(data_cfg.data_root)
    train_days = [
        d for d in all_days if data_cfg.train_start <= d <= data_cfg.train_end
    ]
    val_days = [
        d for d in all_days if data_cfg.val_start <= d <= data_cfg.val_end
    ]
    test_days = [
        d for d in all_days if data_cfg.test_start <= d <= data_cfg.test_end
    ]

    train_loader = _make_loader(train_days, cfg, cache, shuffle=True)
    # Keep val_loader ordered so we can map batches to dataset.index.
    val_loader = _make_loader(val_days, cfg, cache, shuffle=False)
    test_loader = _make_loader(test_days, cfg, cache, shuffle=False)
    return train_loader, val_loader, test_loader


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


def train(
    cfg: SimpleTrainConfig,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Train with logging + early stopping and save artifacts."""
    set_seed(cfg.seed)
    device = cfg.device

    train_loader, val_loader, test_loader = build_loaders(cfg)

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

        val_loss = float("nan")
        val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}
        do_val = (epoch % cfg.val_interval == 0) or (epoch == cfg.max_epochs - 1)
        if do_val:
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

        if do_val:
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

    model.load_state_dict(best_state)
    test_df = collect_predictions(model, test_loader, device)
    test_metrics = compute_group_metrics(test_df)

    model_dir = os.path.join(cfg.save_dir, cfg.model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{cfg.run_name}_best.pt")
    hist_path = os.path.join(model_dir, f"{cfg.run_name}_history.csv")
    metrics_path = os.path.join(model_dir, f"{cfg.run_name}_val_metrics.json")
    test_metrics_path = os.path.join(model_dir, f"{cfg.run_name}_test_metrics.json")

    torch.save(best_state, model_path)
    history_df.to_csv(hist_path, index=False)
    pd.Series(best_val_metrics).to_json(metrics_path, force_ascii=False, indent=2)
    pd.Series(test_metrics).to_json(test_metrics_path, force_ascii=False, indent=2)

    print(f"Saved best model to: {model_path}")
    print(f"Saved history to:    {hist_path}")
    print(f"Saved metrics to:    {metrics_path}")
    print(f"Saved test metrics to: {test_metrics_path}")

    return best_state, history_df, best_val_metrics, test_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = SimpleTrainConfig()
    data_cfg = cfg.data
    parser = argparse.ArgumentParser(description="Run a single simple_train experiment.")
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--hidden_dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--run_name", type=str, default=cfg.run_name)
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--input_dim", type=int, default=cfg.input_dim)
    parser.add_argument("--num_layers", type=int, default=cfg.num_layers)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--max_epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--early_stop_metric", type=str, default=cfg.early_stop_metric)
    parser.add_argument("--val_interval", type=int, default=cfg.val_interval)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--save_dir", type=str, default=cfg.save_dir)
    parser.add_argument("--data_root", type=str, default=data_cfg.data_root)
    parser.add_argument("--stats_path", type=str, default=data_cfg.stats_path)
    parser.add_argument("--window", type=int, default=data_cfg.window)
    parser.add_argument("--cache_size", type=int, default=data_cfg.cache_size)
    parser.add_argument("--num_workers", type=int, default=data_cfg.num_workers)
    parser.add_argument("--train_start", type=str, default=data_cfg.train_start)
    parser.add_argument("--train_end", type=str, default=data_cfg.train_end)
    parser.add_argument("--val_start", type=str, default=data_cfg.val_start)
    parser.add_argument("--val_end", type=str, default=data_cfg.val_end)
    parser.add_argument("--test_start", type=str, default=data_cfg.test_start)
    parser.add_argument("--test_end", type=str, default=data_cfg.test_end)
    return parser


def build_config_from_args(args: argparse.Namespace) -> SimpleTrainConfig:
    data_cfg = DataConfig(
        data_root=args.data_root,
        stats_path=args.stats_path,
        window=args.window,
        cache_size=args.cache_size,
        num_workers=args.num_workers,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    return SimpleTrainConfig(
        model_name=args.model_name,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        val_interval=args.val_interval,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        run_name=args.run_name,
        data=data_cfg,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    train(cfg)
