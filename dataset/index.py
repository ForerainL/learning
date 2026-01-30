"""Index construction for valid (day, skey, t) samples."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .reader import Key

IndexEntry = Tuple[str, str, int]


def build_index(meta: Dict[Key, int], window: int) -> List[IndexEntry]:
    """Generate valid indices and skip the first (window-1) timesteps."""
    if window <= 0:
        raise ValueError("window must be positive")

    min_t = window - 1
    index: List[IndexEntry] = []
    for (day, skey), length in meta.items():
        if length <= min_t:
            continue
        for t in range(min_t, length):
            index.append((day, skey, t))
    return index


__all__ = ["build_index", "IndexEntry"]
