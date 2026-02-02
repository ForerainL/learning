"""Train and evaluate the block-based pipeline with optimizer switching."""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset.block_dataset import BlockDataset, IndexEntry
from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel


@dataclass
class Train2Config:
    tensor_root: str = "./global_data"
    window: int = 64
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    early_stop_metric: str = "IC"  # "val_loss" or "IC"
    val_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./artifacts"
    run_name: str = "train2"

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

    optimizer_name: str = "adamw"  # "adamw" | "sgd" | "sam" | "soap"
    lr: float = 1e-3
    weight_decay: float = 0.01
    momentum: float = 0.9
    sam_rho: float = 0.05
    sam_base: str = "adamw"  # "adamw" | "sgd"
    soap_beta: float = 0.9
    soap_eps: float = 1e-8

    loss_type: str = "mse"  # "mse" | "l1" | "huber" | "linear_combo" | "uncertainty_weighting"
    huber_delta: float = 1.0
    linear_weights: Dict[str, float] = field(default_factory=dict)
    uw_losses: List[str] = field(default_factory=lambda: ["mse", "l1", "huber"])
    uw_huber_delta: float = 1.0


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) wrapper for a base optimizer."""

    def __init__(self, params, base_optimizer_cls, rho: float, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive for SAM")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError("SAM requires first_step and second_step calls")

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norms.append(torch.norm(p.grad.detach(), p=2))
        if not norms:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(norms), p=2)


class SOAP(torch.optim.Optimizer):
    """Simple diagonal preconditioned optimizer with EMA of squared gradients."""

    def __init__(self, params, lr: float, beta: float, eps: float, weight_decay: float):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)

                v = state["v"]
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(grad, denom, value=-lr)


class UncertaintyWeightingLoss(nn.Module):
    """Trainable uncertainty weighting for multiple loss terms."""

    def __init__(self, num_losses: int) -> None:
        super().__init__()
        if num_losses <= 0:
            raise ValueError("num_losses must be positive for UncertaintyWeightingLoss")
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total


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
    device: str,
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


def _validate_linear_weights(weights: Dict[str, float]) -> Dict[str, float]:
    allowed = {"mse", "l1", "huber"}
    unknown = set(weights) - allowed
    if unknown:
        raise ValueError(f"Unknown linear_weights keys: {sorted(unknown)}")
    resolved = {name: float(weights.get(name, 0.0)) for name in allowed}
    if not any(val > 0 for val in resolved.values()):
        raise ValueError("linear_weights must include at least one weight > 0")
    return resolved


def _validate_uw_losses(losses: Sequence[str]) -> List[str]:
    allowed = {"mse", "l1", "huber"}
    if not losses:
        raise ValueError("uw_losses must include at least one loss type")
    normalized = [str(name).lower() for name in losses]
    unknown = set(normalized) - allowed
    if unknown:
        raise ValueError(f"Unknown uw_losses entries: {sorted(unknown)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("uw_losses must not contain duplicates")
    return normalized


def compute_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: Train2Config,
    loss_type: str,
    linear_weights: Optional[Dict[str, float]] = None,
    uw_losses: Optional[Sequence[str]] = None,
    uw_module: Optional[UncertaintyWeightingLoss] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute training loss with selectable modes."""
    components: Dict[str, torch.Tensor] = {}
    if loss_type == "mse":
        loss = F.mse_loss(pred, target)
        return loss, components
    if loss_type == "l1":
        loss = F.l1_loss(pred, target)
        return loss, components
    if loss_type == "huber":
        loss = F.huber_loss(pred, target, delta=cfg.huber_delta)
        return loss, components
    if loss_type == "linear_combo":
        if linear_weights is None:
            raise ValueError("linear_weights must be provided for linear_combo loss")
        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        huber_loss = F.huber_loss(pred, target, delta=cfg.huber_delta)
        components = {"mse": mse_loss, "l1": l1_loss, "huber": huber_loss}
        loss = (
            linear_weights["mse"] * mse_loss
            + linear_weights["l1"] * l1_loss
            + linear_weights["huber"] * huber_loss
        )
        return loss, components
    if loss_type == "uncertainty_weighting":
        if uw_module is None or uw_losses is None:
            raise ValueError("uw_losses and uw_module must be provided for uncertainty_weighting loss")
        loss_map = {
            "mse": F.mse_loss(pred, target),
            "l1": F.l1_loss(pred, target),
            "huber": F.huber_loss(pred, target, delta=cfg.uw_huber_delta),
        }
        losses = [loss_map[name] for name in uw_losses]
        return uw_module(losses), components
    raise ValueError(f"Unknown loss_type: {cfg.loss_type}")


def run_validation_once(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    cfg: Train2Config,
    loss_type: str,
    linear_weights: Optional[Dict[str, float]] = None,
    uw_losses: Optional[Sequence[str]] = None,
    uw_module: Optional[UncertaintyWeightingLoss] = None,
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
            loss, _ = compute_training_loss(
                pred,
                yb,
                cfg,
                loss_type,
                linear_weights=linear_weights,
                uw_losses=uw_losses,
                uw_module=uw_module,
            )
            batch_size = yb.shape[0]
            loss_sum += loss.detach().float().cpu().item() * batch_size
            count += batch_size
            preds.append(pred.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
    val_loss = float("nan") if count == 0 else loss_sum / count
    pred_all = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    y_all = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    return val_loss, compute_overall_metrics(pred_all, y_all)


def build_model(cfg: Train2Config, input_dim: int) -> nn.Module:
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


def build_optimizer(
    model: nn.Module,
    config: Train2Config,
    extra_params: Optional[Sequence[torch.nn.Parameter]] = None,
) -> torch.optim.Optimizer:
    params: List[torch.nn.Parameter] = list(model.parameters())
    if extra_params:
        params.extend(list(extra_params))
    name = (config.optimizer_name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    if name == "sam":
        base_name = config.sam_base.lower()
        if base_name == "sgd":
            base_cls = torch.optim.SGD
            base_kwargs = {
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "momentum": config.momentum,
            }
        elif base_name == "adamw":
            base_cls = torch.optim.AdamW
            base_kwargs = {
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
        else:
            raise ValueError(f"Unknown sam_base: {config.sam_base}")
        return SAM(
            params,
            base_cls,
            rho=config.sam_rho,
            **base_kwargs,
        )
    if name == "soap":
        return SOAP(
            params,
            lr=config.lr,
            beta=config.soap_beta,
            eps=config.soap_eps,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer_name: {config.optimizer_name}")


def train_with_optimizer_switch(
    cfg: Train2Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    starts: List[int],
    meta: List[Tuple[int, str, str]],
) -> None:
    loss_type = cfg.loss_type.lower()
    linear_weights: Optional[Dict[str, float]] = None
    uw_losses: Optional[List[str]] = None
    uw_module: Optional[UncertaintyWeightingLoss] = None

    if loss_type == "linear_combo":
        linear_weights = _validate_linear_weights(cfg.linear_weights)
    elif loss_type == "uncertainty_weighting":
        uw_losses = _validate_uw_losses(cfg.uw_losses)
        uw_module = UncertaintyWeightingLoss(len(uw_losses)).to(cfg.device)
    elif loss_type not in {"mse", "l1", "huber"}:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    extra_params = list(uw_module.parameters()) if uw_module is not None else None
    optimizer = build_optimizer(model, cfg, extra_params=extra_params)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_IC": [],
        "val_RankIC": [],
        "val_QuantileReturn": [],
    }
    if loss_type == "linear_combo":
        history.update(
            {
                "train/mse_loss": [],
                "train/l1_loss": [],
                "train/huber_loss": [],
            }
        )

    best_score = float("inf") if cfg.early_stop_metric == "val_loss" else -float("inf")
    best_state = model.state_dict()
    best_val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "QuantileReturn": float("nan")}
    patience_cnt = 0
    use_sam = isinstance(optimizer, SAM)

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = torch.tensor(0.0, device=cfg.device)
        train_count = 0
        mse_sum = torch.tensor(0.0, device=cfg.device)
        l1_sum = torch.tensor(0.0, device=cfg.device)
        huber_sum = torch.tensor(0.0, device=cfg.device)

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            pred = model(xb).squeeze(-1)
            loss, components = compute_training_loss(
                pred,
                yb,
                cfg,
                loss_type,
                linear_weights=linear_weights,
                uw_losses=uw_losses,
                uw_module=uw_module,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if use_sam:
                optimizer.first_step(zero_grad=True)
                pred = model(xb).squeeze(-1)
                loss, components = compute_training_loss(
                    pred,
                    yb,
                    cfg,
                    loss_type,
                    linear_weights=linear_weights,
                    uw_losses=uw_losses,
                    uw_module=uw_module,
                )
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            train_loss_sum += loss.detach().float() * yb.shape[0]
            train_count += yb.shape[0]
            if loss_type == "linear_combo":
                mse_sum += components["mse"].detach() * yb.shape[0]
                l1_sum += components["l1"].detach() * yb.shape[0]
                huber_sum += components["huber"].detach() * yb.shape[0]

        train_loss = float("nan") if train_count == 0 else (train_loss_sum / train_count).item()

        if epoch % cfg.val_every == 0:
            val_loss, val_metrics = run_validation_once(
                model,
                val_loader,
                cfg.device,
                cfg,
                loss_type,
                linear_weights=linear_weights,
                uw_losses=uw_losses,
                uw_module=uw_module,
            )
        else:
            val_loss = float("nan")
            val_metrics = {"IC": float("nan"), "RankIC": float("nan"), "R2": float("nan"), "QuantileReturn": float("nan")}

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_IC"].append(val_metrics["IC"])
        history["val_RankIC"].append(val_metrics["RankIC"])
        history["val_QuantileReturn"].append(val_metrics["QuantileReturn"])
        if loss_type == "linear_combo":
            if train_count == 0:
                history["train/mse_loss"].append(float("nan"))
                history["train/l1_loss"].append(float("nan"))
                history["train/huber_loss"].append(float("nan"))
            else:
                history["train/mse_loss"].append(float((mse_sum / train_count).item()))
                history["train/l1_loss"].append(float((l1_sum / train_count).item()))
                history["train/huber_loss"].append(float((huber_sum / train_count).item()))

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

    model.load_state_dict(best_state)
    test_df = collect_predictions(model, test_loader, cfg.device, starts, meta)
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


def main() -> None:
    cfg = Train2Config()
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

    train_with_optimizer_switch(cfg, model, train_loader, val_loader, test_loader, starts, meta)


if __name__ == "__main__":
    main()
