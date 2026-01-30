"""DDP-ready dataloader for (day, skey, t_end) index sampling.

Design goals:
- Dataset only slices cached tensors (no IO/normalization per sample).
- LRU cache keeps (day, skey) tensors in memory.
- DistributedSampler handles shuffling for multi-GPU runs.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IndexEntry = Tuple[str, str, int]
TensorPair = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class LoaderConfig:
    data_root: str = "./tensor_data"
    index_path: str = "./index_table.pkl"
    window: int = 64
    cache_size: int = 32

    batch_size: int = 2000
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


def load_index_table(path: str) -> List[IndexEntry]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_day_skey_tensor(data_root: str, day: str, skey: str) -> TensorPair:
    """Load preprocessed tensors for a single (day, skey) without transforms."""
    base = os.path.join(data_root, day, skey)
    if os.path.isfile(f"{base}.pt"):
        payload = torch.load(f"{base}.pt", map_location="cpu")
        return payload["X"], payload["y"]
    x_path = os.path.join(base, "X.pt")
    y_path = os.path.join(base, "y.pt")
    return torch.load(x_path, map_location="cpu"), torch.load(y_path, map_location="cpu")


class DayStockCache:
    """LRU cache for (day, skey) -> (X, y) tensors.

    Note: This cache is per-DataLoader-worker by design in PyTorch. # MODIFIED
    The same (day, skey) may be loaded once per worker, which is expected. # MODIFIED
    """

    def __init__(self, data_root: str, cache_size: int = 32) -> None:
        self.data_root = data_root
        self.cache_size = cache_size
        self._store: "OrderedDict[Tuple[str, str], TensorPair]" = OrderedDict()

    def get(self, day: str, skey: str) -> TensorPair:
        key = (day, skey)
        if key in self._store:
            value = self._store.pop(key)
            self._store[key] = value
            return value
        value = load_day_skey_tensor(self.data_root, day, skey)
        self._store[key] = value
        if len(self._store) > self.cache_size:
            self._store.popitem(last=False)
        return value


class IndexedWindowDataset(Dataset):
    """Dataset that only indexes + slices cached tensors (no IO per sample)."""

    def __init__(
        self,
        index_table: Sequence[IndexEntry],
        window: int,
        cache: DayStockCache,
        return_meta: bool = False,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        # Avoid unnecessary copies when index_table is already a list. # MODIFIED
        self.index_table = index_table if isinstance(index_table, list) else list(index_table)
        self.window = window
        self.cache = cache
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, idx: int):
        day, skey, t_end = self.index_table[idx]
        x_all, y_all = self.cache.get(day, skey)
        x = x_all[t_end - self.window : t_end]
        y = y_all[t_end - 1]
        if self.return_meta:
            return x, y, day, skey
        return x, y


def count_unique_day_skey(batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]) -> int:
    """Debug hook: count unique (day, skey) in a batch (requires return_meta=True)."""
    _, _, day_list, skey_list = batch
    pairs = set(zip(day_list, skey_list))
    return len(pairs)


def build_ddp_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,  # MODIFIED: ensure each rank has aligned batch counts
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return loader, sampler


def init_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def run_example(rank: int, world_size: int, cfg: LoaderConfig) -> None:
    init_distributed(rank, world_size)

    index_table = load_index_table(cfg.index_path)
    cache = DayStockCache(cfg.data_root, cache_size=cfg.cache_size)
    dataset = IndexedWindowDataset(index_table, cfg.window, cache, return_meta=True)

    loader, sampler = build_ddp_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        rank=rank,
        world_size=world_size,
    )

    # MODIFIED: sampler.set_epoch(epoch) should be called inside the training loop.
    for step, batch in enumerate(loader):
        if step >= 1:
            break
        unique_pairs = count_unique_day_skey(batch)
        x, y, _, _ = batch
        print(
            f"rank={rank} | batch={step} | x={tuple(x.shape)} y={tuple(y.shape)} | "
            f"unique(day,skey)={unique_pairs}"
        )

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP dataloader example.")
    parser.add_argument("--data-root", default="./tensor_data")
    parser.add_argument("--index-path", default="./index_table.pkl")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    cfg = LoaderConfig(
        data_root=args.data_root,
        index_path=args.index_path,
        window=args.window,
        cache_size=args.cache_size,
        batch_size=args.batch_size,
    )

    torch.multiprocessing.spawn(
        run_example,
        args=(args.world_size, cfg),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
