"""Submission-row construction and mechanical validation."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .rle import decode_rle_counts, encode_binary_mask

_FILAMENT_ID = re.compile(r"^(?P<image_id>.+)_(?P<instance>[1-9][0-9]*)$")


@dataclass(frozen=True)
class SubmissionRow:
    """One predicted filament instance."""

    filament_id: str
    segmentation_rle: str


def rows_for_image(image_id: str, masks: Sequence[ArrayLike]) -> list[SubmissionRow]:
    """Encode all predicted instances for one test-image stem."""
    clean_image_id = image_id.strip()
    if not clean_image_id:
        raise ValueError("image_id must be non-empty.")
    return [
        SubmissionRow(
            filament_id=f"{clean_image_id}_{index}",
            segmentation_rle=encode_binary_mask(mask),
        )
        for index, mask in enumerate(masks, start=1)
    ]


def validate_submission_rows(
    rows: Iterable[SubmissionRow],
    *,
    known_image_ids: set[str] | None = None,
    height: int = 2048,
    width: int = 2048,
    decode_masks: bool = True,
) -> list[SubmissionRow]:
    """Validate identifiers and masks without using hidden test labels."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("A submission must contain at least one predicted filament.")

    seen: set[str] = set()
    masks_by_image: dict[str, set[str]] = {}
    for row in materialized:
        if row.filament_id in seen:
            raise ValueError(f"Duplicate filament_id: {row.filament_id}")
        seen.add(row.filament_id)

        match = _FILAMENT_ID.fullmatch(row.filament_id)
        if match is None:
            raise ValueError(f"Invalid filament_id: {row.filament_id}")
        image_id = match.group("image_id")
        if known_image_ids is not None and image_id not in known_image_ids:
            raise ValueError(f"Unknown test image ID in {row.filament_id}.")

        counts = row.segmentation_rle
        if not counts or not counts.isascii():
            raise ValueError(f"Invalid compressed RLE for {row.filament_id}.")
        if any(character in counts for character in ('"', "'", "\r", "\n")):
            raise ValueError(f"RLE contains a forbidden quote or newline for {row.filament_id}.")
        image_masks = masks_by_image.setdefault(image_id, set())
        if counts in image_masks:
            raise ValueError(f"Duplicate instance mask in image {image_id}.")
        image_masks.add(counts)

        if decode_masks:
            try:
                decoded = decode_rle_counts(counts, height=height, width=width)
            except (TypeError, ValueError) as error:
                raise ValueError(f"RLE cannot be decoded for {row.filament_id}.") from error
            if not np.any(decoded):
                raise ValueError(f"Empty instance mask for {row.filament_id}.")
            if encode_binary_mask(decoded) != counts:
                raise ValueError(f"RLE is not a canonical round-trip for {row.filament_id}.")
    return materialized


def write_submission(
    rows: Iterable[SubmissionRow],
    path: str | Path,
    *,
    known_image_ids: set[str] | None = None,
    height: int = 2048,
    width: int = 2048,
) -> Path:
    """Write a mechanically validated competition CSV."""
    materialized = validate_submission_rows(
        rows,
        known_image_ids=known_image_ids,
        height=height,
        width=width,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows((row.filament_id, row.segmentation_rle) for row in materialized)
    return output_path
