"""Minimal DDP training skeleton using IndexedWindowDataset + DayStockCache."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataset.ddp_loader import (
    DayStockCache,
    IndexedWindowDataset,
    build_ddp_dataloader,
    load_index_table,
)


@dataclass
class TrainConfig:
    data_root: str = "./tensor_data"
    index_path: str = "./index_table.pkl"
    window: int = 64
    cache_size: int = 32

    batch_size: int = 2000
    epochs: int = 2
    lr: float = 1e-3


def init_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def build_model(feature_dim: int, window: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(window * feature_dim, 1),
    )


def run_worker(rank: int, world_size: int, cfg: TrainConfig) -> None:
    init_distributed(rank, world_size)

    index_table = load_index_table(cfg.index_path)
    cache = DayStockCache(cfg.data_root, cache_size=cfg.cache_size)
    dataset = IndexedWindowDataset(index_table, cfg.window, cache, return_meta=False)
    loader, sampler = build_ddp_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        rank=rank,
        world_size=world_size,
    )

    # Infer feature dimension from the first (day, skey).
    day, skey, _ = index_table[0]
    x_all, _ = cache.get(day, skey)
    feature_dim = x_all.shape[1]

    model = build_model(feature_dim, cfg.window).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        for xb, yb in loader:
            xb = xb.to(rank, non_blocking=True)
            yb = yb.to(rank, non_blocking=True)

            pred = ddp_model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"epoch={epoch} | last_loss={loss.item():.6f}")

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal DDP training example.")
    parser.add_argument("--data-root", default="./tensor_data")
    parser.add_argument("--index-path", default="./index_table.pkl")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        index_path=args.index_path,
        window=args.window,
        cache_size=args.cache_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    torch.multiprocessing.spawn(
        run_worker,
        args=(args.world_size, cfg),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
