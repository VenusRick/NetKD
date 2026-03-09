"""Fallback implementations of common classification metrics.

These utilities cover the subset of ``sklearn.metrics`` APIs that NetKD uses,
allowing the project to run in environments where scikit-learn is unavailable.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _prep_arrays(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    labels = sorted({int(x) for x in np.concatenate([true, pred])})
    return true, pred, labels


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true, pred, _ = _prep_arrays(y_true, y_pred)
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    true, pred, labels = _prep_arrays(y_true, y_pred)
    size = len(labels)
    matrix = np.zeros((size, size), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for t, p in zip(true, pred):
        matrix[label_to_idx[int(t)], label_to_idx[int(p)]] += 1
    return matrix


def precision_recall_fscore_support(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str | None = None,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    true, pred, labels = _prep_arrays(y_true, y_pred)
    if true.size == 0:
        zeros = np.zeros(len(labels), dtype=float)
        return zeros, zeros, zeros, zeros

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for label in labels:
        mask_true = true == label
        mask_pred = pred == label
        tp = float(np.sum(mask_true & mask_pred))
        fp = float(np.sum(~mask_true & mask_pred))
        fn = float(np.sum(mask_true & ~mask_pred))
        support = float(np.sum(mask_true))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    precisions_arr = np.array(precisions, dtype=float)
    recalls_arr = np.array(recalls, dtype=float)
    f1s_arr = np.array(f1s, dtype=float)
    supports_arr = np.array(supports, dtype=float)

    if average == "weighted":
        total = supports_arr.sum() or 1.0
        weights = supports_arr / total
        return (
            float(np.sum(precisions_arr * weights)),
            float(np.sum(recalls_arr * weights)),
            float(np.sum(f1s_arr * weights)),
            float(total),
        )

    return precisions_arr, recalls_arr, f1s_arr, supports_arr


def f1_score(y_true: Sequence[int], y_pred: Sequence[int], average: str = "weighted") -> float:
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    return float(f1)


__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_recall_fscore_support",
]
