"""Parquet reader that normalizes features before window slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch

Key = Tuple[str, str]


@dataclass(frozen=True)
class _GroupStats:
    cols: List[str]
    # Only a subset of fields are used per group.
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    median: torch.Tensor | None = None
    mad: torch.Tensor | None = None
    clip: torch.Tensor | None = None
    scale: torch.Tensor | None = None


def _as_group_stats(group: Mapping[str, object]) -> _GroupStats:
    return _GroupStats(
        cols=list(group.get("cols", [])),
        mean=group.get("mean"),
        std=group.get("std"),
        median=group.get("median"),
        mad=group.get("mad"),
        clip=group.get("clip"),
        scale=group.get("scale"),
    )


class DayReader:
    """Read a (day, skey) parquet and apply training-only normalization."""

    def __init__(self, data_root: str, stats_path: str = "feature_stats.pt") -> None:
        self.data_root = data_root
        raw_stats: Dict[str, Mapping[str, object]] = torch.load(stats_path)
        self.smooth = _as_group_stats(raw_stats["smooth"])
        self.heavy = _as_group_stats(raw_stats["heavy"])
        self.discrete = _as_group_stats(raw_stats["discrete"])
        self.feature_cols: List[str] = (
            self.smooth.cols + self.heavy.cols + self.discrete.cols
        )

    def _parquet_path(self, day: str, skey: str) -> str:
        return f"{self.data_root}/{day}/{skey}.parquet"

    @staticmethod
    def _to_tensor(frame: pd.DataFrame, cols: Sequence[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((len(frame), 0), dtype=torch.float32)
        return torch.tensor(frame.loc[:, cols].to_numpy(), dtype=torch.float32)

    @staticmethod
    def _normalize_smooth(x: torch.Tensor, stats: _GroupStats) -> torch.Tensor:
        if x.numel() == 0:
            return x
        assert stats.mean is not None and stats.std is not None
        mean = stats.mean.to(dtype=x.dtype, device=x.device)
        std = stats.std.to(dtype=x.dtype, device=x.device)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (x - mean) / std

    @staticmethod
    def _normalize_heavy(x: torch.Tensor, stats: _GroupStats) -> torch.Tensor:
        if x.numel() == 0:
            return x
        assert stats.median is not None and stats.mad is not None and stats.clip is not None
        median = stats.median.to(dtype=x.dtype, device=x.device)
        mad = stats.mad.to(dtype=x.dtype, device=x.device)
        mad = torch.where(mad == 0, torch.ones_like(mad), mad)
        clip = stats.clip.to(dtype=x.dtype, device=x.device)
        normalized = (x - median) / mad
        return torch.clamp(normalized, min=-clip, max=clip)

    @staticmethod
    def _normalize_discrete(x: torch.Tensor, stats: _GroupStats) -> torch.Tensor:
        if x.numel() == 0:
            return x
        assert stats.scale is not None
        scale = stats.scale.to(dtype=x.dtype, device=x.device)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return x / scale

    def read(self, day: str, skey: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return normalized features [T, D] and raw labels [T]."""
        path = self._parquet_path(day, skey)
        frame = pd.read_parquet(path, columns=self.feature_cols + ["label"])

        xs = self._to_tensor(frame, self.smooth.cols)
        xh = self._to_tensor(frame, self.heavy.cols)
        xd = self._to_tensor(frame, self.discrete.cols)

        xs = self._normalize_smooth(xs, self.smooth)
        xh = self._normalize_heavy(xh, self.heavy)
        xd = self._normalize_discrete(xd, self.discrete)

        # Normalization must happen before any window slicing.
        x = torch.cat([xs, xh, xd], dim=1).to(dtype=torch.float32)
        y = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
        return x, y


__all__ = ["DayReader", "Key"]
