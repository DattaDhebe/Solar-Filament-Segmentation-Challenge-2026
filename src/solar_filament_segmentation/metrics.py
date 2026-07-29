"""Low-level binary-mask diagnostics, not a full organizer evaluator replica."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _binary_pair(
    prediction: ArrayLike,
    target: ArrayLike,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"Mask shapes differ: prediction={predicted.shape}, target={expected.shape}."
        )
    if predicted.ndim != 2:
        raise ValueError(f"Binary mask diagnostics require 2D arrays, got {predicted.ndim}D.")
    return predicted.astype(bool, copy=False), expected.astype(bool, copy=False)


def binary_dice(prediction: ArrayLike, target: ArrayLike, *, empty_score: float = 1.0) -> float:
    """Compute Sørensen-Dice for one already-matched pair of binary masks."""
    predicted, expected = _binary_pair(prediction, target)
    denominator = int(predicted.sum()) + int(expected.sum())
    if denominator == 0:
        return float(empty_score)
    intersection = int(np.logical_and(predicted, expected).sum())
    return float((2 * intersection) / denominator)


def binary_iou(prediction: ArrayLike, target: ArrayLike, *, empty_score: float = 1.0) -> float:
    """Compute intersection-over-union for one already-matched binary-mask pair."""
    predicted, expected = _binary_pair(prediction, target)
    union = int(np.logical_or(predicted, expected).sum())
    if union == 0:
        return float(empty_score)
    intersection = int(np.logical_and(predicted, expected).sum())
    return float(intersection / union)
