"""Validate downloaded experiment-004 metadata and every submission RLE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from solar_filament_segmentation.submission import (
    SubmissionRow,
    validate_submission_rows,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-004-component-quality.yaml"
EXPECTED_FOLD_FINGERPRINT = "69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7"


def parse_arguments() -> argparse.Namespace:
    """Parse one downloaded Kaggle output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--test-image-directory",
        type=Path,
        help="Optional official test JPEG directory for exact image-ID validation.",
    )
    parser.add_argument(
        "--require-promotion-gate",
        action="store_true",
        help="Fail unless the predeclared grouped-OOF promotion gate passed.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[SubmissionRow]:
    """Read exactly the required two CSV columns."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected submission columns: {reader.fieldnames}")
        return [
            SubmissionRow(
                filament_id=record["filament_id"],
                segmentation_rle=record["segmentation_rle"],
            )
            for record in reader
        ]


def validate_metadata(metadata: dict, *, require_promotion_gate: bool) -> None:
    """Enforce provenance, fold, selection, and finite-metric invariants."""
    expected_hash = hashlib.sha256(
        CONFIG_PATH.read_text(encoding="utf-8").strip().encode()
    ).hexdigest()
    required_equalities = {
        "experiment_id": "experiment-004",
        "config_sha256": expected_hash,
        "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
        "checkpoint_source": "experiment-003",
        "checkpoint_training_enabled": False,
        "external_labeled_data": False,
        "pretrained_weights": False,
        "test_statistics_used_for_selection": False,
    }
    for key, expected in required_equalities.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Invalid metadata {key}: expected {expected!r}, got {metadata.get(key)!r}."
            )

    diagnostics = metadata.get("selected_oof_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Missing selected_oof_diagnostics.")
    for key in (
        "mean_penalized_instance_dice",
        "mean_matched_instance_dice",
        "mean_matched_instance_iou",
        "missed",
        "extra",
    ):
        value = diagnostics.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Invalid selected OOF diagnostic {key}: {value!r}.")

    gate = metadata.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ValueError("Missing or invalid promotion_gate.")
    if require_promotion_gate and not gate["passed"]:
        raise ValueError("Experiment 004 did not pass its predeclared grouped-OOF promotion gate.")


def main() -> None:
    """Validate downloaded run metadata and submission candidate."""
    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    metadata_path = output_directory / "experiment-004-run-metadata.json"
    submission_path = output_directory / "experiment-004-submission.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(
        metadata,
        require_promotion_gate=arguments.require_promotion_gate,
    )

    known_image_ids = None
    if arguments.test_image_directory is not None:
        known_image_ids = {
            path.stem for path in arguments.test_image_directory.resolve().glob("*.jpeg")
        }
        if not known_image_ids:
            raise ValueError("The supplied test-image directory contains no JPEGs.")

    rows = read_rows(submission_path)
    validate_submission_rows(
        rows,
        known_image_ids=known_image_ids,
        height=2048,
        width=2048,
        decode_masks=True,
    )
    if len(rows) != metadata.get("submission_rows"):
        raise ValueError("CSV row count disagrees with run metadata.")

    observed_counts = Counter(row.filament_id.rpartition("_")[0] for row in rows)
    expected_counts = metadata.get("predicted_instances_per_image")
    if not isinstance(expected_counts, dict):
        raise ValueError("Missing predicted_instances_per_image metadata.")
    for image_id, expected_count in expected_counts.items():
        if observed_counts[image_id] != expected_count:
            raise ValueError(f"Instance count disagrees for test image {image_id}.")

    diagnostics = metadata["selected_oof_diagnostics"]
    print(f"Validated experiment 004 output: {submission_path}")
    print(f"Submission rows: {len(rows)}")
    print(
        "OOF diagnostics: "
        f"penalized Dice={diagnostics['mean_penalized_instance_dice']:.6f}, "
        f"matched Dice={diagnostics['mean_matched_instance_dice']:.6f}, "
        f"matched IoU={diagnostics['mean_matched_instance_iou']:.6f}, "
        f"missed={diagnostics['missed']}, extra={diagnostics['extra']}"
    )
    print(f"Promotion gate passed: {metadata['promotion_gate']['passed']}")
    print("All masks are non-empty, unique, canonical 2048x2048 COCO compressed RLE.")


if __name__ == "__main__":
    main()
