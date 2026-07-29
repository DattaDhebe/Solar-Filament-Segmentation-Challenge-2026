"""Read-only structural inspection of the downloaded competition data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from solar_filament_segmentation.data import (
    load_coco_annotations,
    physical_observation_key,
    summarize_coco,
)

_OBSERVATION_STEM = re.compile(r"^[0-9]{14}[A-Za-z]{2}$")


def file_sha256(path: Path) -> str:
    """Hash one local competition image without persisting image content."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_inventory(directory: Path) -> dict[str, Any]:
    """Inspect image headers, identifiers, and exact-byte duplicates."""
    image_paths = sorted(directory.glob("*.jpeg"))
    sizes: set[tuple[int, int]] = set()
    modes: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.add(image.size)
            modes.add(image.mode)
        hashes[file_sha256(image_path)].append(image_path.name)
    return {
        "count": len(image_paths),
        "sizes": sorted([list(size) for size in sizes]),
        "modes": sorted(modes),
        "stems": {path.stem for path in image_paths},
        "hashes": hashes,
        "years": dict(sorted(Counter(path.stem[:4] for path in image_paths).items())),
        "instruments": dict(sorted(Counter(path.stem[-2:] for path in image_paths).items())),
        "invalid_names": [
            path.name for path in image_paths if _OBSERVATION_STEM.fullmatch(path.stem) is None
        ],
    }


def annotation_inventory(document: dict[str, Any], train_stems: set[str]) -> dict[str, Any]:
    """Validate COCO identity and single-polygon invariants."""
    images = document["images"]
    annotations = document["annotations"]
    categories = {category["id"]: category["name"] for category in document["categories"]}
    image_ids = [record["id"] for record in images]
    annotation_ids = [record["id"] for record in annotations]
    known_image_ids = set(image_ids)
    physical_keys = [physical_observation_key(record) for record in images]
    records_per_observation = Counter(physical_keys)
    category_counts = Counter(categories[record["category_id"]] for record in annotations)

    invalid_polygons = 0
    multi_polygon_annotations = 0
    nonzero_iscrowd = 0
    invalid_bounding_boxes = 0
    nonpositive_areas = 0
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            invalid_polygons += 1
            continue
        if len(segmentation) != 1:
            multi_polygon_annotations += 1
        polygon = segmentation[0]
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2
            or any(
                not isinstance(coordinate, int | float)
                or not math.isfinite(coordinate)
                or coordinate < 0
                or coordinate > 2048
                for coordinate in polygon
            )
        ):
            invalid_polygons += 1
        if annotation.get("iscrowd") != 0:
            nonzero_iscrowd += 1
        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, int | float) or not math.isfinite(value) for value in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= 0
            or bbox[3] <= 0
            or bbox[0] + bbox[2] > 2048.01
            or bbox[1] + bbox[3] > 2048.01
        ):
            invalid_bounding_boxes += 1
        if not isinstance(annotation.get("area"), int | float) or annotation["area"] <= 0:
            nonpositive_areas += 1

    return {
        "image_records": len(images),
        "physical_observations": len(records_per_observation),
        "records_per_observation": dict(sorted(Counter(records_per_observation.values()).items())),
        "duplicate_annotation_sets": len(images) - len(records_per_observation),
        "annotation_records": len(annotations),
        "category_counts": dict(sorted(category_counts.items())),
        "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
        "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
        "orphan_annotations": sum(
            annotation["image_id"] not in known_image_ids for annotation in annotations
        ),
        "missing_training_images": sorted(set(physical_keys) - train_stems),
        "unreferenced_training_images": sorted(train_stems - set(physical_keys)),
        "invalid_coco_image_sizes": sum(
            record.get("height") != 2048 or record.get("width") != 2048 for record in images
        ),
        "invalid_polygons": invalid_polygons,
        "multi_polygon_annotations": multi_polygon_annotations,
        "nonzero_iscrowd": nonzero_iscrowd,
        "invalid_bounding_boxes": invalid_bounding_boxes,
        "nonpositive_areas": nonpositive_areas,
        "info": document["info"],
        "licenses": document["licenses"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/setup/data-audit.json"),
    )
    args = parser.parse_args()

    annotation_path = args.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    document = load_coco_annotations(annotation_path)
    summary = summarize_coco(document)
    train = image_inventory(args.data_root / "train" / "train_images")
    test = image_inventory(args.data_root / "test" / "test_images")
    annotations = annotation_inventory(document, train["stems"])

    train_test_stem_overlap = sorted(train["stems"] & test["stems"])
    train_test_hash_overlap = sorted(set(train["hashes"]) & set(test["hashes"]))
    duplicate_train_hashes = {
        digest: names for digest, names in train["hashes"].items() if len(names) > 1
    }
    duplicate_test_hashes = {
        digest: names for digest, names in test["hashes"].items() if len(names) > 1
    }

    audit = {
        "annotation": annotations,
        "train_images": {
            key: value for key, value in train.items() if key not in {"stems", "hashes"}
        },
        "test_images": {
            key: value for key, value in test.items() if key not in {"stems", "hashes"}
        },
        "leakage_checks": {
            "train_test_stem_overlap": train_test_stem_overlap,
            "train_test_exact_hash_overlap": train_test_hash_overlap,
            "duplicate_train_hashes": duplicate_train_hashes,
            "duplicate_test_hashes": duplicate_test_hashes,
        },
    }

    print(f"COCO image records: {summary.image_records}")
    print(f"Physical annotated observations: {summary.physical_observations}")
    print(f"Duplicate annotator image records: {summary.duplicate_annotation_sets}")
    print(f"Filament annotations: {summary.annotation_records}")
    print(f"Categories: {', '.join(summary.categories)}")
    print(f"Annotation sets per observation: {annotations['records_per_observation']}")
    print(f"Category counts: {annotations['category_counts']}")
    print(f"Training JPEGs: {train['count']}; sizes={train['sizes']}; modes={train['modes']}")
    print(f"Test JPEGs: {test['count']}; sizes={test['sizes']}; modes={test['modes']}")
    print(
        "Train/test overlap: "
        f"stems={len(train_test_stem_overlap)}, exact_hashes={len(train_test_hash_overlap)}"
    )
    print(
        "Duplicate image bytes: "
        f"train={len(duplicate_train_hashes)}, test={len(duplicate_test_hashes)}"
    )

    if train["count"] != 707 or test["count"] != 180:
        raise SystemExit("Unexpected competition image inventory.")
    if train["sizes"] != [[2048, 2048]] or test["sizes"] != [[2048, 2048]]:
        raise SystemExit("Unexpected image dimensions.")
    if train["modes"] != ["L"] or test["modes"] != ["L"]:
        raise SystemExit("Competition images are not consistently decoded as grayscale.")
    if train_test_stem_overlap or train_test_hash_overlap:
        raise SystemExit("Exact train/test image overlap detected.")
    if duplicate_train_hashes or duplicate_test_hashes:
        raise SystemExit("Duplicate image bytes detected within a competition split.")
    structural_failures = {
        key: annotations[key]
        for key in (
            "duplicate_image_ids",
            "duplicate_annotation_ids",
            "orphan_annotations",
            "missing_training_images",
            "unreferenced_training_images",
            "invalid_coco_image_sizes",
            "invalid_polygons",
            "multi_polygon_annotations",
            "nonzero_iscrowd",
            "invalid_bounding_boxes",
            "nonpositive_areas",
        )
        if annotations[key]
    }
    if structural_failures:
        raise SystemExit(f"Annotation structure validation failed: {structural_failures}")
    if train["invalid_names"] or test["invalid_names"]:
        raise SystemExit("Unexpected observation filename format.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Audit written outside Git: {args.output}")


if __name__ == "__main__":
    main()
