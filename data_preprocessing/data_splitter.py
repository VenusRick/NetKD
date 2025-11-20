"""Data splitting utilities for SD-MKD preprocessing."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import random

try:
    from sklearn.model_selection import KFold as SklearnKFold, StratifiedKFold as SklearnStratifiedKFold  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SklearnKFold = None
    SklearnStratifiedKFold = None


class _KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data: Sequence[int], labels: Sequence[int] | None = None):  # noqa: D401
        n = len(data)
        indices = list(range(n))
        rng = random.Random(self.random_state)
        if self.shuffle:
            rng.shuffle(indices)
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = indices[:start] + indices[stop:]
            yield train_idx, val_idx
            current = stop


class _StratifiedKFold(_KFold):
    def split(self, data: Sequence[int], labels: Sequence[int]):
        n = len(data)
        rng = random.Random(self.random_state)
        buckets: dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            buckets.setdefault(int(label), []).append(idx)

        folds: List[List[int]] = [[] for _ in range(self.n_splits)]
        for bucket in buckets.values():
            if self.shuffle:
                rng.shuffle(bucket)
            for i, idx in enumerate(bucket):
                folds[i % self.n_splits].append(idx)

        all_indices = set(range(n))
        for fold in folds:
            val_idx = sorted(fold)
            train_idx = sorted(all_indices - set(val_idx))
            yield train_idx, val_idx


KFold = SklearnKFold if SklearnKFold is not None else _KFold
StratifiedKFold = SklearnStratifiedKFold if SklearnStratifiedKFold is not None else _StratifiedKFold


def train_val_test_split(indices: Sequence[int], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split indices into train/val/test subsets."""

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    n_total = len(indices)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]
    return train_idx, val_idx, test_idx


def stacking_split(indices: Sequence[int], stacking_ratio: float = 0.3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split training indices into base-model and stacking subsets."""

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    split = int(len(indices) * (1 - stacking_ratio))
    return indices[:split], indices[split:]


def generate_kfold_splits(indices: Sequence[int], labels: Sequence[int], k: int = 5, stratified: bool = True, seed: int = 42):
    """Yield k-fold train/val index pairs, optionally stratified."""

    if stratified:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(indices, labels):
            yield np.array(indices)[train_idx], np.array(indices)[val_idx]
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(indices):
            yield np.array(indices)[train_idx], np.array(indices)[val_idx]


__all__ = ["train_val_test_split", "stacking_split", "generate_kfold_splits"]
