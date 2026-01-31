"""DDP-capable training entry based on train1.py with minimal changes."""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset.block_dataset import BlockDataset, IndexEntry
from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel


@dataclass
class TrainDDPConfig:
    tensor_root: str = "./global_data"
    window: int = 64
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    val_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./artifacts"
    run_name: str = "train_ddp"

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

    use_ddp: bool = True
    ddp_min_global_batch: int = 16384
    num_workers: int = 0
    seed: int = 42


def is_ddp() -> bool:
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return get_rank() == 0


def ddp_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_stop_flag(stop: bool) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return stop
    tensor = torch.tensor(1 if stop else 0, device=torch.device("cuda") if torch.cuda.is_available() else "cpu")
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    if not is_ddp():
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = get_world_size()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return True, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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


def compute_overall_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute overall validation metrics without grouping."""
    if len(pred) == 0:
        return {"IC": float("nan"), "RankIC": float("nan"), "R2": float("nan"), "QuantileReturn": float("nan")}
    pred_series = pd.Series(pred)
    y_series = pd.Series(y)
    ic = float(pred_series.corr(y_series))
    rank_ic = float(pred_series.rank().corr(y_series.rank()))
    y_mean = float(y_series.mean())
    denom = float(((y_series - y_mean) ** 2).sum())
    r2 = float("nan") if denom == 0 else 1.0 - float(((y_series - pred_series) ** 2).sum()) / denom
    order = np.argsort(pred)
    q = max(int(0.1 * len(pred)), 1)
    qret = float(y_series.iloc[order[-q:]].mean() - y_series.iloc[order[:q]].mean())
    return {"IC": ic, "RankIC": rank_ic, "R2": r2, "QuantileReturn": qret}


def build_index_from_segments(
    segments: Sequence[Dict[str, object]],
    window: int,
    start: str,
    end: str,
) -> List[IndexEntry]:
    index: List[IndexEntry] = []
    for seg in segments:
        day = str(seg["day"])
        if not (start <= day <= end):
            continue
        seg_start = int(seg["start_idx"])
        length = int(seg["length"])
        if length <= window:
            continue
        index.extend(range(seg_start + window, seg_start + length))
    return index


def build_segment_ranges(
    segments: Sequence[Dict[str, object]],
) -> Tuple[List[int], List[Tuple[int, str, str]]]:
    ranges = []
    for seg in segments:
        start = int(seg["start_idx"])
        end = start + int(seg["length"])
        ranges.append((start, end, str(seg["day"]), str(seg["skey"])))
    ranges.sort(key=lambda r: r[0])
    starts = [r[0] for r in ranges]
    meta = [(r[1], r[2], r[3]) for r in ranges]
    return starts, meta


def find_segment_meta(
    t_end: int, starts: List[int], meta: List[Tuple[int, str, str]]
) -> Tuple[str, str]:
    idx = int(np.searchsorted(starts, t_end, side="right") - 1)
    if idx < 0:
        raise IndexError("t_end before any segment start.")
    end, day, skey = meta[idx]
    if t_end > end:
        raise IndexError("t_end exceeds segment end.")
    return day, skey


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    starts: List[int],
    meta: List[Tuple[int, str, str]],
) -> pd.DataFrame:
    model.eval()
    dataset: BlockDataset = loader.dataset  # type: ignore[assignment]
    index_table = dataset.index_table
    rows: List[dict] = []
    offset = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).squeeze(-1).detach().cpu().numpy()
        y = yb.detach().cpu().numpy()

        batch_size = len(y)
        batch_index = index_table[offset : offset + batch_size]
        offset += batch_size

        for i, t_end in enumerate(batch_index):
            day, skey = find_segment_meta(int(t_end), starts, meta)
            rows.append(
                {
                    "date": day,
                    "stock": skey,
                    "y": float(y[i]),
                    "pred": float(pred[i]),
                }
            )

    return pd.DataFrame(rows)


def run_validation_once(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    preds: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            batch_size = yb.shape[0]
            loss_sum += loss.detach().float().cpu().item() * batch_size
            count += batch_size
            preds.append(pred.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
    val_loss = float("nan") if count == 0 else loss_sum / count
    pred_all = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    y_all = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    return val_loss, compute_overall_metrics(pred_all, y_all)


def build_model(cfg: TrainDDPConfig, input_dim: int) -> nn.Module:
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


def run_once(cfg: TrainDDPConfig) -> None:
    torch.manual_seed(cfg.seed)
    ddp_enabled, rank, local_rank, world_size, device = setup_distributed()
    global_batch = cfg.batch_size * (world_size if ddp_enabled else 1)
    use_ddp_train = cfg.use_ddp and ddp_enabled and world_size > 1 and global_batch >= cfg.ddp_min_global_batch

    if ddp_enabled and not use_ddp_train and not is_rank0():
        ddp_barrier()
        cleanup_distributed()
        return

    if is_rank0():
        os.makedirs(cfg.save_dir, exist_ok=True)

    segment_table_path = os.path.join(cfg.tensor_root, "segment_table.pkl")
    with open(segment_table_path, "rb") as f:
        segments = pickle.load(f)

    train_index = build_index_from_segments(segments, cfg.window, cfg.train_start, cfg.train_end)
    val_index = build_index_from_segments(segments, cfg.window, cfg.val_start, cfg.val_end)
    test_index = build_index_from_segments(segments, cfg.window, cfg.test_start, cfg.test_end)
    starts, meta = build_segment_ranges(segments)

    train_ds = BlockDataset(train_index, cfg.tensor_root, cfg.window)
    val_ds = BlockDataset(val_index, cfg.tensor_root, cfg.window)
    test_ds = BlockDataset(test_index, cfg.tensor_root, cfg.window)

    train_sampler = None
    if use_ddp_train:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = None
    test_loader = None
    if is_rank0():
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    feature_dim = train_ds.x_all.shape[1]
    model = build_model(cfg, feature_dim).to(device)
    if use_ddp_train:
        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = DDP(
                model,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

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
    best_state = None
    best_val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}
    patience_cnt = 0

    for epoch in range(cfg.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach().float() * yb.shape[0]
            train_count += yb.shape[0]

        train_loss = float("nan") if train_count == 0 else (train_loss_sum / train_count).item()

        val_loss = float("nan")
        val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "R2": float("nan"), "QuantileReturn": float("nan")}
        if is_rank0() and val_loader is not None and epoch % cfg.val_every == 0:
            val_loss, val_metrics = run_validation_once(model, val_loader, device, loss_fn)

        if is_rank0():
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

        stop = False
        if is_rank0() and epoch % cfg.val_every == 0:
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
                    stop = True

        if use_ddp_train:
            stop = broadcast_stop_flag(stop)
        if stop:
            if is_rank0():
                print(f"Early stopping at epoch {epoch}")
            break

    if is_rank0():
        if best_state is None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        test_df = collect_predictions(model, test_loader, device, starts, meta) if test_loader else pd.DataFrame()
        test_metrics = compute_group_metrics(test_df) if not test_df.empty else {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}

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

    if ddp_enabled:
        ddp_barrier()
        cleanup_distributed()


def main() -> None:
    cfg = TrainDDPConfig()
    run_once(cfg)


if __name__ == "__main__":
    main()
