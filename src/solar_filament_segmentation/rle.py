"""COCO polygon and compressed-RLE conversion helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pycocotools import mask as mask_utils


def _binary_2d(mask: ArrayLike) -> NDArray[np.uint8]:
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}.")
    return (binary != 0).astype(np.uint8, copy=False)


def encode_binary_mask(mask: ArrayLike) -> str:
    """Encode one binary mask as the ASCII COCO compressed `counts` string."""
    binary = np.asfortranarray(_binary_2d(mask))
    encoded = mask_utils.encode(binary)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str) or not counts:
        raise ValueError("pycocotools returned an invalid compressed-RLE count string.")
    return counts


def decode_rle_counts(counts: str, *, height: int = 2048, width: int = 2048) -> NDArray[np.uint8]:
    """Decode a COCO compressed `counts` string for a known mask size."""
    if not isinstance(counts, str) or not counts:
        raise ValueError("counts must be a non-empty string.")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    decoded = mask_utils.decode(
        {
            "size": [height, width],
            "counts": counts.encode("ascii"),
        }
    )
    return _binary_2d(decoded)


def polygons_to_mask(
    polygons: Sequence[Sequence[float]],
    *,
    height: int = 2048,
    width: int = 2048,
) -> NDArray[np.uint8]:
    """Rasterize one or more COCO polygons into one binary mask."""
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_utils.frPyObjects(polygons, height, width)
    merged = mask_utils.merge(rles)
    return _binary_2d(mask_utils.decode(merged))
