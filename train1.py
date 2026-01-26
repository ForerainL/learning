"""Train and evaluate the block-based pipeline end-to-end."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.block_dataset import BlockDataset, BlockEntry, load_block_table
from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel


@dataclass
class Train1Config:
    tensor_root: str = "./global_data"
    block_table_path: str = "./global_data/block_table.pkl"
    window: int = 64
    block_len: int = 512
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./artifacts"
    run_name: str = "train1"

    train_start: str = "20220101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240331"
    test_start: str = "20240401"
    test_end: str = "20241231"

    model_name: str = "gru"
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.1


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(values))


def compute_group_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """IC, RankIC, Quantile return per (date, stock), averaged."""
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


def _filter_blocks(
    blocks: Sequence[BlockEntry], start: str, end: str
) -> List[BlockEntry]:
    return [b for b in blocks if start <= str(b["day"]) <= end]


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str) -> pd.DataFrame:
    model.eval()
    dataset: BlockDataset = loader.dataset  # type: ignore[assignment]
    blocks = dataset.block_table
    rows: List[dict] = []
    offset = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).squeeze(-1).detach().cpu().numpy()
        y = yb.detach().cpu().numpy()

        batch_size = len(y)
        batch_blocks = blocks[offset : offset + batch_size]
        offset += batch_size

        for i, block in enumerate(batch_blocks):
            rows.append(
                {
                    "date": block["day"],
                    "stock": block["skey"],
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
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def build_model(cfg: Train1Config, input_dim: int) -> nn.Module:
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
    cfg = Train1Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    block_table = load_block_table(cfg.block_table_path)
    train_blocks = _filter_blocks(block_table, cfg.train_start, cfg.train_end)
    val_blocks = _filter_blocks(block_table, cfg.val_start, cfg.val_end)
    test_blocks = _filter_blocks(block_table, cfg.test_start, cfg.test_end)

    train_ds = BlockDataset(train_blocks, cfg.tensor_root, cfg.window, cfg.block_len)
    val_ds = BlockDataset(val_blocks, cfg.tensor_root, cfg.window, cfg.block_len)
    test_ds = BlockDataset(test_blocks, cfg.tensor_root, cfg.window, cfg.block_len)

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
    test_loader = DataLoader(
        test_ds,
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
        train_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss = compute_loss(model, val_loader, cfg.device, loss_fn)
        val_df = collect_predictions(model, val_loader, cfg.device)
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

    model.load_state_dict(best_state)
    test_df = collect_predictions(model, test_loader, cfg.device)
    test_metrics = compute_group_metrics(test_df)

    history_df = pd.DataFrame(history)
    model_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
    hist_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_history.csv")
    val_metrics_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_val_metrics.json")
    test_metrics_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_test_metrics.json")

    torch.save(best_state, model_path)
    history_df.to_csv(hist_path, index=False)
    pd.Series(best_val_metrics).to_json(val_metrics_path, force_ascii=False, indent=2)
    pd.Series(test_metrics).to_json(test_metrics_path, force_ascii=False, indent=2)

    print(f"Saved best model to: {model_path}")
    print(f"Saved history to:    {hist_path}")
    print(f"Saved metrics to:    {val_metrics_path}")
    print(f"Saved test metrics to: {test_metrics_path}")


if __name__ == "__main__":
    main()
