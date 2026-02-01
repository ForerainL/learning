"""Train and validate with day-shuffled split (train/val only)."""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.block_dataset import BlockDataset, IndexEntry
from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel
from train1 import build_segment_ranges, run_validation_once


@dataclass
class TrainShuffleByDayConfig:
    tensor_root: str = "./global_data"
    window: int = 64
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    val_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./artifacts"
    run_name: str = "train_shuffle_by_day"

    model_name: str = "gru"
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.1

    train_ratio: float = 0.8
    val_ratio: float = 0.2
    split_seed: int = 42


def build_index_from_segments_by_day(
    segments: Sequence[Dict[str, object]],
    window: int,
    days: Set[str],
) -> List[IndexEntry]:
    index: List[IndexEntry] = []
    for seg in segments:
        day = str(seg["day"])
        if day not in days:
            continue
        seg_start = int(seg["start_idx"])
        length = int(seg["length"])
        if length <= window:
            continue
        index.extend(range(seg_start + window, seg_start + length))
    return index


def build_model(cfg: TrainShuffleByDayConfig, input_dim: int) -> nn.Module:
    model_cfg = ModelConfig(
        input_dim=input_dim,
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


def main() -> None:
    cfg = TrainShuffleByDayConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)
    assert abs(cfg.train_ratio + cfg.val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must be 1.0"

    segment_table_path = os.path.join(cfg.tensor_root, "segment_table.pkl")
    with open(segment_table_path, "rb") as f:
        segments = pickle.load(f)

    # Here we shuffle days once and split into train/val.
    all_days = sorted({str(seg["day"]) for seg in segments})
    rng = np.random.default_rng(cfg.split_seed)
    shuffled_days = list(all_days)
    rng.shuffle(shuffled_days)
    split_idx = int(len(shuffled_days) * cfg.train_ratio)
    train_days = set(shuffled_days[:split_idx])
    val_days = set(shuffled_days[split_idx:])

    print(f"#train_days = {len(train_days)}")
    print(f"#val_days   = {len(val_days)}")
    assert train_days.isdisjoint(val_days), "train_days and val_days must be disjoint"

    train_index = build_index_from_segments_by_day(segments, cfg.window, train_days)
    val_index = build_index_from_segments_by_day(segments, cfg.window, val_days)
    starts, meta = build_segment_ranges(segments)

    train_ds = BlockDataset(train_index, cfg.tensor_root, cfg.window)
    val_ds = BlockDataset(val_index, cfg.tensor_root, cfg.window)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    feature_dim = train_ds.x_all.shape[1]
    model = build_model(cfg, feature_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
    best_state = model.state_dict()
    best_val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}
    patience_cnt = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = torch.tensor(0.0, device=cfg.device)
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach().float() * yb.shape[0]
            train_count += yb.shape[0]

        train_loss = float("nan") if train_count == 0 else (train_loss_sum / train_count).item()

        if epoch % cfg.val_every == 0:
            val_loss, val_metrics = run_validation_once(model, val_loader, cfg.device, loss_fn)
        else:
            val_loss = float("nan")
            val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "R2": float("nan"), "QuantileReturn": float("nan")}

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

        if epoch % cfg.val_every == 0:
            score = val_loss if cfg.early_stop_metric == "val_loss" else val_metrics["IC"]
            improved = (score < best_score) if cfg.early_stop_metric == "val_loss" else (score > best_score)
            if not math.isnan(score) and improved:
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

    model_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
    hist_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_history.csv")
    val_metrics_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_val_metrics.json")

    torch.save(best_state, model_path)
    history_df.to_csv(hist_path, index=False)
    pd.Series(best_val_metrics).to_json(val_metrics_path, force_ascii=False, indent=2)

    print(f"Saved best model to: {model_path}")
    print(f"Saved history to:    {hist_path}")
    print(f"Saved metrics to:    {val_metrics_path}")


if __name__ == "__main__":
    main()
