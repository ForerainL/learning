"""Simple LRU cache around a DayReader."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch

from .reader import DayReader, Key

Value = Tuple[torch.FloatTensor, torch.FloatTensor]


class DayCache:
    """Cache normalized (x, y) per (day, skey) to avoid repeated IO."""

    def __init__(self, reader: DayReader, max_size: int = 128) -> None:
        self.reader = reader
        self.max_size = max_size
        self._store: "OrderedDict[Key, Value]" = OrderedDict()

    def _touch(self, key: Key, value: Value) -> Value:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)
        return value

    def get(self, day: str, skey: str) -> Value:
        key: Key = (day, skey)
        if key in self._store:
            value = self._store.pop(key)
            return self._touch(key, value)
        value = self.reader.read(day, skey)
        return self._touch(key, value)


__all__ = ["DayCache", "Value"]
