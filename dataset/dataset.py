"""Dataset that only slices tensors; no IO or normalization here."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .cache import DayCache
from .index import IndexEntry


class TimeSeriesDataset(Dataset):
    """Serve fixed windows and last-step labels from cached day tensors."""

    def __init__(self, index: Sequence[IndexEntry], cache: DayCache, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.index = list(index)
        self.cache = cache
        self.window = window

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        day, skey, t = self.index[idx]
        x_day, y_day = self.cache.get(day, skey)

        start = t - self.window + 1
        # build_index guarantees start >= 0, so no padding is needed.
        x = x_day[start : t + 1]
        y = y_day[t]
        return x, y


__all__ = ["TimeSeriesDataset"]
