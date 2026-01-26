"""Block-based dataset for fast window sampling from global tensors."""

from __future__ import annotations

import os
import pickle
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

IndexEntry = int


class BlockDataset(Dataset):
    """Sample windows from a precomputed index table without per-sample IO.

    This is fast because X_all/y_all are loaded once in memory and
    each sample only slices tensors. Epochs iterate over the explicit
    window index list rather than the full tick-level dataset.
    """

    def __init__(
        self,
        index_table: Sequence[IndexEntry],
        tensor_root: str,
        window: int,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.index_table = list(index_table)
        self.tensor_root = tensor_root
        self.window = window

        # Load global tensors once (no per-sample IO).
        self.x_all = torch.load(os.path.join(tensor_root, "X_all.pt"), map_location="cpu")
        self.y_all = torch.load(os.path.join(tensor_root, "y_all.pt"), map_location="cpu")

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t_end = self.index_table[idx]
        x = self.x_all[t_end - self.window : t_end]
        y = self.y_all[t_end - 1]
        return x, y


def load_index_table(path: str) -> List[IndexEntry]:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = ["BlockDataset", "load_index_table", "IndexEntry"]
