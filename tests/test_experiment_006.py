import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_006.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-006-oof-blend-calibration.yaml"


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


def _load_kernel_module():
    stubbed_cv2 = "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None
    if stubbed_cv2:
        sys.modules["cv2"] = types.ModuleType("cv2")
    specification = importlib.util.spec_from_file_location("experiment_006", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_006_embeds_the_versioned_config() -> None:
    embedded_text = _embedded_config_yaml()
    versioned_text = CONFIG_PATH.read_text(encoding="utf-8").strip()

    assert embedded_text == versioned_text
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-006"


def test_calibration_grid_is_frozen_and_complete() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    candidates = module.calibration_candidates(config)

    assert len(candidates) == 135
    assert {record["experiment_005_probability_weight"] for record in candidates} == {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert any(
        module.candidate_matches(
            record,
            config["calibration"]["reference_candidates"]["experiment-003"],
        )
        for record in candidates
    )
    assert any(
        module.candidate_matches(
            record,
            config["calibration"]["reference_candidates"]["experiment-005"],
        )
        for record in candidates
    )


def test_only_competition_checkpoint_sources_are_enabled() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["model"]["competition_trained_checkpoint_sources"] == [
        "experiment-003",
        "experiment-005",
    ]
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_probability_blend_has_fixed_endpoint_behavior() -> None:
    module = _load_kernel_module()
    experiment_003 = [np.array([[0.2, 0.4]], dtype=np.float32)]
    experiment_005 = [np.array([[0.6, 0.8]], dtype=np.float32)]

    first = list(
        module.blended_probabilities(
            experiment_003,
            experiment_005,
            experiment_005_weight=0.0,
        )
    )
    middle = list(
        module.blended_probabilities(
            experiment_003,
            experiment_005,
            experiment_005_weight=0.25,
        )
    )
    last = list(
        module.blended_probabilities(
            experiment_003,
            experiment_005,
            experiment_005_weight=1.0,
        )
    )

    np.testing.assert_array_equal(first[0], experiment_003[0])
    np.testing.assert_allclose(middle[0], np.array([[0.3, 0.5]], dtype=np.float32))
    np.testing.assert_array_equal(last[0], experiment_005[0])


def test_selection_prefers_predeclared_feasible_candidates() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    results = [
        {
            "instance": {
                "mean_penalized_instance_dice": 0.51,
                "mean_matched_instance_dice": 0.73,
                "missed": 1600,
                "extra": 2200,
            }
        },
        {
            "instance": {
                "mean_penalized_instance_dice": 0.50,
                "mean_matched_instance_dice": 0.72,
                "missed": 1650,
                "extra": 2050,
            }
        },
    ]

    selected = module.select_calibration_candidate(results, config=config)

    assert selected is results[1]
    assert selected["feasible"] is True
    assert results[0]["feasible"] is False


def test_promotion_gate_requires_all_three_improvements() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())

    passed = module.blend_promotion_result(
        {"mean_penalized_instance_dice": 0.5, "missed": 1650, "extra": 2050},
        config=config,
    )
    failed = module.blend_promotion_result(
        {"mean_penalized_instance_dice": 0.5, "missed": 1650, "extra": 2200},
        config=config,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["extra_instances"] is False


def test_oof_evidence_is_written_before_test_images_are_loaded() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)

    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection': False" in source
