"""Validate downloaded experiment-005 OOF evidence and submission artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from solar_filament_segmentation.submission import SubmissionRow, validate_submission_rows

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-005-hybrid-annotator.yaml"


def parse_arguments() -> argparse.Namespace:
    """Parse one unique downloaded Kaggle output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--test-image-directory", type=Path)
    parser.add_argument(
        "--oof-only",
        action="store_true",
        help="Validate the pre-test OOF checkpoint when a run did not finish.",
    )
    parser.add_argument(
        "--require-promotion-gate",
        action="store_true",
        help="Fail unless the predeclared full-OOF promotion gate passed.",
    )
    return parser.parse_args()


def validate_oof_metadata(metadata: dict, *, require_promotion_gate: bool) -> None:
    """Validate experiment provenance, selection isolation, and finite metrics."""
    config_text = CONFIG_PATH.read_text(encoding="utf-8").strip()
    config = yaml.safe_load(config_text)
    expected = {
        "experiment_id": "experiment-005",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "fold_fingerprint": config["validation"]["expected_fold_fingerprint"],
        "checkpoint_source": "experiment-003",
        "external_labeled_data": False,
        "pretrained_weights": False,
        "test_statistics_used_for_selection": False,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Invalid {key}: expected {expected_value!r}, got {metadata.get(key)!r}."
            )

    selected = metadata.get("selected_weight_pair")
    if selected not in config["screen"]["weight_pairs"]:
        raise ValueError(f"Selected weight pair was not predeclared: {selected!r}.")
    folds = metadata.get("folds")
    if not isinstance(folds, list) or sorted(record["fold"] for record in folds) != list(range(5)):
        raise ValueError("Metadata must contain exactly one selected result for all five folds.")
    summary = metadata.get("full_oof")
    if not isinstance(summary, dict):
        raise ValueError("Missing full_oof diagnostics.")
    for key in (
        "mean_penalized_instance_dice",
        "mean_matched_instance_dice",
        "mean_matched_instance_iou",
        "missed",
        "extra",
    ):
        value = summary.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Invalid full_oof {key}: {value!r}.")
    gate = metadata.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ValueError("Missing or invalid promotion gate.")
    if require_promotion_gate and not gate["passed"]:
        raise ValueError("Experiment 005 did not pass its predeclared promotion gate.")


def read_rows(path: Path) -> list[SubmissionRow]:
    """Read exactly the required CSV schema."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        return [
            SubmissionRow(record["filament_id"], record["segmentation_rle"]) for record in reader
        ]


def main() -> None:
    """Validate OOF-only evidence or a completed candidate."""
    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    metadata_name = (
        "experiment-005-oof-metadata.json"
        if arguments.oof_only
        else "experiment-005-run-metadata.json"
    )
    metadata = json.loads((output_directory / metadata_name).read_text(encoding="utf-8"))
    validate_oof_metadata(
        metadata,
        require_promotion_gate=arguments.require_promotion_gate,
    )
    summary = metadata["full_oof"]
    print(
        "Validated experiment 005 OOF evidence: "
        f"Dice={summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={summary['missed']}, extra={summary['extra']}, "
        f"promotion={metadata['promotion_gate']['passed']}"
    )
    if arguments.oof_only:
        return

    known_ids = None
    if arguments.test_image_directory is not None:
        known_ids = {path.stem for path in arguments.test_image_directory.resolve().glob("*.jpeg")}
        if not known_ids:
            raise ValueError("No JPEGs found in the test-image directory.")
    rows = read_rows(output_directory / "experiment-005-submission.csv")
    validate_submission_rows(
        rows,
        known_image_ids=known_ids,
        height=2048,
        width=2048,
        decode_masks=True,
    )
    if len(rows) != metadata.get("submission_rows"):
        raise ValueError("Submission row count disagrees with metadata.")
    observed = Counter(row.filament_id.rpartition("_")[0] for row in rows)
    expected = metadata.get("predicted_instances_per_image")
    if not isinstance(expected, dict):
        raise ValueError("Missing predicted_instances_per_image.")
    for image_id, count in expected.items():
        if observed[image_id] != count:
            raise ValueError(f"Instance count mismatch for {image_id}.")
    print(f"Validated {len(rows)} unique, non-empty, canonical 2048x2048 RLE masks.")


if __name__ == "__main__":
    main()
