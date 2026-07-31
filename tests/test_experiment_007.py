import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_007.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-007-boundary-seeded-unet.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_007", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_007_embeds_the_versioned_config() -> None:
    embedded_text = _embedded_config_yaml()
    versioned_text = CONFIG_PATH.read_text(encoding="utf-8").strip()

    assert embedded_text == versioned_text
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-007"


def test_boundary_model_uses_only_competition_checkpoint_supervision() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["model"]["competition_trained_checkpoint_source"] == "experiment-003"
    assert config["model"]["heads"] == [
        "foreground",
        "boundary",
        "normalized-instance-distance",
    ]
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_boundary_model_source_load_declares_only_new_heads_missing() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    build_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_model"
    )
    train_fold = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_fold"
    )
    model_source = ast.unparse(build_model)
    training_source = ast.unparse(train_fold)

    assert "self.head" in model_source
    assert "self.boundary_head" in model_source
    assert "self.distance_head" in model_source
    assert "strict=False" in training_source
    assert "boundary_head.weight" in training_source
    assert "distance_head.weight" in training_source


def test_frozen_candidate_grid_contains_control_and_96_seeded_settings() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    candidates = module.postprocessing_candidates(config)

    assert len(candidates) == 97
    assert sum(candidate["method"] == "connected-components" for candidate in candidates) == 1
    assert sum(candidate["method"] == "seeded-partition" for candidate in candidates) == 96
    assert all(candidate["minimum_instance_area"] in {64, 96} for candidate in candidates[1:])


def test_annotator_sampling_remains_deterministic_and_epoch_seeded() -> None:
    module = _load_kernel_module()

    first = [module.deterministic_annotator_index(31, epoch, 7, 3) for epoch in range(12)]
    second = [module.deterministic_annotator_index(31, epoch, 7, 3) for epoch in range(12)]

    assert first == second
    assert len(set(first)) > 1
    assert all(0 <= value < 3 for value in first)


def test_labeled_instance_filter_is_deterministic_and_confidence_aware() -> None:
    module = _load_kernel_module()
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:4, 1:5] = 1
    labels[7:10, 7:11] = 2
    foreground = np.zeros((12, 12), dtype=np.float32)
    foreground[labels == 1] = 0.9
    foreground[labels == 2] = 0.6

    instances = module.filter_labeled_instances(
        labels,
        foreground,
        minimum_area=8,
        minimum_mean_probability=0.8,
        maximum_area=100,
    )

    assert len(instances) == 1
    np.testing.assert_array_equal(instances[0], labels == 1)


def test_selection_prefers_a_feasible_boundary_candidate() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    results = [
        {
            "candidate": {"method": "seeded-partition"},
            "evaluation": {
                "instance": {
                    "mean_penalized_instance_dice": 0.51,
                    "mean_matched_instance_dice": 0.73,
                    "missed": 1700,
                    "extra": 2000,
                }
            },
        },
        {
            "candidate": {"method": "seeded-partition"},
            "evaluation": {
                "instance": {
                    "mean_penalized_instance_dice": 0.50,
                    "mean_matched_instance_dice": 0.72,
                    "missed": 1650,
                    "extra": 2050,
                }
            },
        },
    ]

    selected = module.select_candidate(results, config=config)

    assert selected is results[1]
    assert selected["feasible"] is True
    assert results[0]["feasible"] is False


def test_promotion_requires_seeded_method_and_relation_limits() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    evaluation = {
        "instance": {
            "mean_penalized_instance_dice": 0.51,
            "missed": 1650,
            "extra": 2000,
            "one_to_many": 500,
            "many_to_one": 110,
        }
    }

    passed = module.promotion_result(
        {"candidate": {"method": "seeded-partition"}, "evaluation": evaluation},
        config=config,
    )
    failed = module.promotion_result(
        {"candidate": {"method": "connected-components"}, "evaluation": evaluation},
        config=config,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["selected_method"] is False


def test_oof_selection_is_written_before_test_images_are_loaded() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)

    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection': False" in source
