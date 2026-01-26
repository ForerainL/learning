"""Block-based dataset for fast window sampling from global tensors."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

BlockEntry = Dict[str, int | str]


class BlockDataset(Dataset):
    """Sample windows from block_table without per-sample IO.

    This is fast because X_all/y_all are loaded once in memory and
    each sample only slices tensors. Epochs iterate over blocks rather
    than the full tick-level dataset.
    """

    def __init__(
        self,
        block_table: Sequence[BlockEntry],
        tensor_root: str,
        window: int,
        block_len: int,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if block_len <= window:
            raise ValueError("block_len must be larger than window")
        self.block_table = list(block_table)
        self.tensor_root = tensor_root
        self.window = window
        self.block_len = block_len

        # Load global tensors once (no per-sample IO).
        self.x_all = torch.load(os.path.join(tensor_root, "X_all.pt"), map_location="cpu")
        self.y_all = torch.load(os.path.join(tensor_root, "y_all.pt"), map_location="cpu")

        with open(os.path.join(tensor_root, "segment_table.pkl"), "rb") as f:
            segments = pickle.load(f)
        self._segment_map: Dict[Tuple[str, str], Tuple[int, int]] = {
            (row["day"], row["skey"]): (row["start_idx"], row["length"])
            for row in segments
        }

    def __len__(self) -> int:
        return len(self.block_table)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.block_table[idx]
        day = block["day"]
        skey = block["skey"]
        segment_start = int(block["segment_start_idx"])
        block_start = int(block["block_start"])

        _, segment_len = self._segment_map[(day, skey)]
        block_end = min(block_start + self.block_len, segment_len)

        low = segment_start + block_start + self.window
        high = segment_start + block_end
        if high <= low:
            raise IndexError("Block too short for the configured window.")

        # Randomly sample a window end within the block without crossing segment boundaries.
        t_end = int(torch.randint(low, high, (1,)).item())
        x = self.x_all[t_end - self.window : t_end]
        y = self.y_all[t_end - 1]
        return x, y


def load_block_table(path: str) -> List[BlockEntry]:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = ["BlockDataset", "load_block_table", "BlockEntry"]
