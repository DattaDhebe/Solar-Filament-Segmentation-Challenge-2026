"""Small, explicit configuration loader for versioned experiment YAML files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates a project invariant."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate competition-specific invariants without inventing model abstractions."""
    required_sections = {
        "experiment",
        "competition",
        "data",
        "validation",
        "model",
        "metric",
        "submission",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ConfigError(f"Missing top-level configuration sections: {missing}")

    experiment = _mapping(config["experiment"], "experiment")
    if not str(experiment.get("id", "")).strip():
        raise ConfigError("experiment.id must be non-empty.")
    _positive_int(experiment.get("seed"), "experiment.seed")

    competition = _mapping(config["competition"], "competition")
    if competition.get("slug") != "filament-segmentation-2026":
        raise ConfigError("competition.slug must be filament-segmentation-2026.")
    if (
        _positive_int(competition.get("image_height"), "competition.image_height") != 2048
        or _positive_int(competition.get("image_width"), "competition.image_width") != 2048
    ):
        raise ConfigError("Competition masks must be exactly 2048 by 2048 pixels.")
    if competition.get("grayscale") is not True:
        raise ConfigError("GONG H-alpha JPEGs must be configured as grayscale.")

    data = _mapping(config["data"], "data")
    for key in (
        "local_root",
        "kaggle_root",
        "train_images",
        "train_annotations",
        "test_images",
    ):
        if not str(data.get(key, "")).strip():
            raise ConfigError(f"data.{key} must be non-empty.")

    validation = _mapping(config["validation"], "validation")
    if validation.get("group_key") != "file_name":
        raise ConfigError(
            "validation.group_key must be file_name to keep identical observations in one fold."
        )
    if _positive_int(validation.get("n_splits"), "validation.n_splits") < 2:
        raise ConfigError("validation.n_splits must be at least 2.")
    _positive_int(validation.get("seed"), "validation.seed")

    model = _mapping(config["model"], "model")
    if model.get("task") != "instance-segmentation":
        raise ConfigError("model.task must be instance-segmentation.")
    if model.get("external_labeled_data") is not False:
        raise ConfigError(
            "External labeled data is disabled by the competition evaluation wording."
        )

    metric = _mapping(config["metric"], "metric")
    if metric.get("primary") != "mean-matched-instance-dice":
        raise ConfigError("metric.primary must be mean-matched-instance-dice.")

    submission = _mapping(config["submission"], "submission")
    if submission.get("columns") != ["filament_id", "segmentation_rle"]:
        raise ConfigError(
            "submission.columns must be ['filament_id', 'segmentation_rle'] in that order."
        )
    if submission.get("rle_format") != "coco-compressed-counts":
        raise ConfigError("submission.rle_format must be coco-compressed-counts.")
    if (
        _positive_int(submission.get("mask_height"), "submission.mask_height") != 2048
        or _positive_int(submission.get("mask_width"), "submission.mask_width") != 2048
    ):
        raise ConfigError("Submission masks must be exactly 2048 by 2048 pixels.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one experiment YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    config = dict(_mapping(loaded, "configuration"))
    validate_config(config)
    return config
