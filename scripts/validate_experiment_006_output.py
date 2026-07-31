"""Validate downloaded experiment-006 OOF evidence and submission artifacts."""

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
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-006-oof-blend-calibration.yaml"


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
    """Validate provenance, reference regressions, selection, and diagnostics."""
    config_text = CONFIG_PATH.read_text(encoding="utf-8").strip()
    config = yaml.safe_load(config_text)
    expected = {
        "experiment_id": "experiment-006",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "fold_fingerprint": config["validation"]["expected_fold_fingerprint"],
        "checkpoint_sources": ["experiment-003", "experiment-005"],
        "external_labeled_data": False,
        "pretrained_weights": False,
        "test_statistics_used_for_selection": False,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Invalid {key}: expected {expected_value!r}, got {metadata.get(key)!r}."
            )

    calibration = config["calibration"]
    expected_candidates = (
        len(calibration["experiment_005_probability_weights"])
        * len(calibration["probability_thresholds"])
        * len(calibration["closing_kernels"])
        * len(calibration["minimum_component_areas"])
        * len(calibration["minimum_component_mean_probabilities"])
    )
    results = metadata.get("candidate_results")
    if metadata.get("candidate_count") != expected_candidates:
        raise ValueError("Candidate count disagrees with the frozen calibration grid.")
    if not isinstance(results, list) or len(results) != expected_candidates:
        raise ValueError("Calibration results do not cover the complete frozen grid.")

    references = metadata.get("reference_checks")
    if not isinstance(references, dict):
        raise ValueError("Missing reference candidate checks.")
    for source in ("experiment-003", "experiment-005"):
        if references.get(source, {}).get("passed") is not True:
            raise ValueError(f"Reference regression did not pass for {source}.")

    selected = metadata.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("Missing selected calibration candidate.")
    postprocessing = selected.get("postprocessing", {})
    selected_values = {
        "weight": selected.get("experiment_005_probability_weight"),
        "threshold": postprocessing.get("probability_threshold"),
        "closing": postprocessing.get("closing_kernel"),
        "area": postprocessing.get("min_component_area_at_model_resolution"),
        "confidence": postprocessing.get("minimum_component_mean_probability"),
    }
    allowed = {
        "weight": calibration["experiment_005_probability_weights"],
        "threshold": calibration["probability_thresholds"],
        "closing": calibration["closing_kernels"],
        "area": calibration["minimum_component_areas"],
        "confidence": calibration["minimum_component_mean_probabilities"],
    }
    for key, value in selected_values.items():
        if value not in allowed[key]:
            raise ValueError(f"Selected {key} was not predeclared: {value!r}.")

    summary = metadata.get("full_oof")
    if not isinstance(summary, dict):
        raise ValueError("Missing full_oof diagnostics.")
    for key in (
        "mean_penalized_instance_dice",
        "mean_matched_instance_dice",
        "mean_matched_instance_iou",
        "missed",
        "extra",
        "one_to_many",
        "many_to_one",
    ):
        value = summary.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Invalid full_oof {key}: {value!r}.")
    gate = metadata.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ValueError("Missing or invalid promotion gate.")
    if require_promotion_gate and not gate["passed"]:
        raise ValueError("Experiment 006 did not pass its predeclared promotion gate.")


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
        "experiment-006-oof-metadata.json"
        if arguments.oof_only
        else "experiment-006-run-metadata.json"
    )
    metadata = json.loads((output_directory / metadata_name).read_text(encoding="utf-8"))
    validate_oof_metadata(
        metadata,
        require_promotion_gate=arguments.require_promotion_gate,
    )
    summary = metadata["full_oof"]
    selected = metadata["selected_candidate"]
    print(
        "Validated experiment 006 OOF evidence: "
        f"Dice={summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={summary['missed']}, extra={summary['extra']}, "
        f"w005={selected['experiment_005_probability_weight']:.3f}, "
        f"promotion={metadata['promotion_gate']['passed']}"
    )
    if arguments.oof_only:
        return

    known_ids = None
    if arguments.test_image_directory is not None:
        known_ids = {path.stem for path in arguments.test_image_directory.resolve().glob("*.jpeg")}
        if not known_ids:
            raise ValueError("No JPEGs found in the test-image directory.")
    rows = read_rows(output_directory / "experiment-006-submission.csv")
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
    expected_counts = metadata.get("predicted_instances_per_image")
    if not isinstance(expected_counts, dict):
        raise ValueError("Missing predicted_instances_per_image.")
    for image_id, count in expected_counts.items():
        if observed[image_id] != count:
            raise ValueError(f"Instance count mismatch for {image_id}.")
    print(f"Validated {len(rows)} unique, non-empty, canonical 2048x2048 RLE masks.")


if __name__ == "__main__":
    main()
