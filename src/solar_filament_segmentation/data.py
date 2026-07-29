"""COCO annotation inventory helpers that do not load test labels or fit transforms."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoInventory:
    """Compact metadata summary for a MAGFiLO COCO annotation file."""

    image_records: int
    physical_observations: int
    annotation_records: int
    duplicate_annotation_sets: int
    categories: tuple[str, ...]


def load_coco_annotations(path: str | Path) -> dict[str, Any]:
    """Load a COCO-style JSON document and verify its required collections."""
    annotation_path = Path(path)
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    if not isinstance(document, dict):
        raise ValueError("COCO annotation root must be an object.")

    required = {"info", "images", "annotations", "licenses", "categories"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"COCO annotation file is missing collections: {missing}")
    if not isinstance(document["images"], list) or not isinstance(document["annotations"], list):
        raise ValueError("COCO images and annotations must be lists.")
    return document


def physical_observation_key(image_record: dict[str, Any]) -> str:
    """Return the JPEG stem used to group duplicate annotator views in validation."""
    file_name = image_record.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Every COCO image record must contain a non-empty file_name.")
    return Path(file_name).stem


def summarize_coco(document: dict[str, Any]) -> CocoInventory:
    """Summarize annotation-set duplication without inspecting competition test data."""
    images = document["images"]
    annotations = document["annotations"]
    counts = Counter(physical_observation_key(record) for record in images)
    categories = tuple(
        str(category["name"])
        for category in sorted(document["categories"], key=lambda item: item["id"])
    )
    return CocoInventory(
        image_records=len(images),
        physical_observations=len(counts),
        annotation_records=len(annotations),
        duplicate_annotation_sets=sum(count - 1 for count in counts.values()),
        categories=categories,
    )
