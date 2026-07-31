import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_004.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-004-component-quality.yaml"


def _load_kernel_module():
    stubbed_cv2 = "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None
    if stubbed_cv2:
        sys.modules["cv2"] = types.ModuleType("cv2")
    specification = importlib.util.spec_from_file_location("experiment_004", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def _embedded_config_yaml() -> str:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EMBEDDED_CONFIG_YAML"
            for target in node.targets
        )
    )
    strip_call = assignment.value
    assert isinstance(strip_call, ast.Call)
    assert isinstance(strip_call.func, ast.Attribute)
    assert strip_call.func.attr == "strip"
    return str(ast.literal_eval(strip_call.func.value)).strip()


def test_experiment_004_embeds_the_versioned_config() -> None:
    embedded = yaml.safe_load(_embedded_config_yaml())
    versioned = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert embedded == versioned
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-004"


def test_experiment_004_uses_only_frozen_competition_checkpoints() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["model"]["competition_trained_checkpoint_source"] == "experiment-003"
    assert config["model"]["checkpoint_training_enabled"] is False
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["validation"]["component_ranker_cross_fit_by"] == "validation_fold"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_component_features_are_deterministic_and_finite() -> None:
    module = _load_kernel_module()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:24, 12:38] = 1

    first = module.principal_geometry(mask)
    second = module.principal_geometry(mask)

    np.testing.assert_allclose(first[:3], second[:3])
    np.testing.assert_allclose(first[3], second[3])
    assert first[4] == second[4]
    assert np.all(np.isfinite(np.asarray(first[:3])))
    assert np.all(np.isfinite(first[3]))
    assert len(module.FEATURE_NAMES) == 20


def test_component_ranker_cross_fit_rejects_group_leakage(monkeypatch) -> None:
    module = _load_kernel_module()

    class MeanRanker:
        def fit(self, features, targets):
            self.mean = float(np.mean(targets))
            return self

        def predict(self, features):
            return np.full(len(features), self.mean)

    monkeypatch.setattr(module, "make_ranker", lambda _config: MeanRanker())
    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    targets = np.asarray([0.1, 0.2, 0.8, 0.9])
    folds = np.asarray([0, 0, 1, 1])
    leaked_keys = np.asarray(["shared", "fold0", "shared", "fold1"], dtype=object)

    with pytest.raises(RuntimeError, match="leaked"):
        module.cross_fitted_scores(
            features,
            targets,
            folds,
            leaked_keys,
            fold_indices=[0, 1],
            config={},
        )


def test_sparse_instance_diagnostic_matches_dense_version() -> None:
    module = _load_kernel_module()
    first = np.zeros((12, 12), dtype=np.uint8)
    first[1:4, 1:5] = 1
    second = np.zeros((12, 12), dtype=np.uint8)
    second[7:10, 7:11] = 1
    targets = np.zeros((12, 12), dtype=np.uint16)
    targets[1:4, 1:5] = 1
    targets[7:10, 7:11] = 2
    sparse = [
        {"mask": first, "pixel_indices": np.flatnonzero(first)},
        {"mask": second, "pixel_indices": np.flatnonzero(second)},
    ]

    dense_result = module.instance_diagnostic([first, second], targets, minimum_iou=0.1)
    sparse_result = module.sparse_instance_diagnostic(
        sparse,
        targets,
        minimum_iou=0.1,
    )

    assert sparse_result == dense_result


def test_fragment_linking_does_not_cross_disk_boundary() -> None:
    module = _load_kernel_module()
    disk = np.ones((24, 24), dtype=bool)
    disk[:, 10:14] = False

    allowed = module.segment_stays_inside_disk(
        (6, 12),
        (17, 12),
        disk,
        line_thickness=1,
    )

    assert allowed is False
