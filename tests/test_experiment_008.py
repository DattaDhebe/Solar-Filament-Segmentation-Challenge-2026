import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_008.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-008-capacity-resolution.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_008", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_008_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-008"


def test_capacity_resolution_screen_is_frozen_and_from_scratch() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["initialization"] == "from-scratch"
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["screen"]["folds"] == [0, 1]
    assert config["screen"]["settings"] == [
        {"id": "1024-base32", "input_size": 1024, "base_channels": 32},
        {"id": "1280-base24", "input_size": 1280, "base_channels": 24},
    ]
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_train_fold_has_no_checkpoint_load_and_uses_setting_capacity() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    train_fold = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_fold"
    )
    source = ast.unparse(train_fold)
    assert "build_model(setting['base_channels'])" in source
    assert "from-scratch" in source
    assert "resolve_checkpoint" not in source


def test_selection_prefers_semantic_gate_then_instance_dice() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    results = [
        {
            "setting": config["screen"]["settings"][0],
            "mean_penalized_instance_dice": 0.52,
            "mean_semantic_dice": 0.68,
            "semantic_gate_passed": False,
        },
        {
            "setting": config["screen"]["settings"][1],
            "mean_penalized_instance_dice": 0.50,
            "mean_semantic_dice": 0.67,
            "semantic_gate_passed": True,
        },
    ]
    selected, selected_id = module.select_setting(results, config=config)
    assert selected_id == "1280-base24"
    assert selected == config["screen"]["settings"][1]


def test_semantic_gate_reads_each_screen_fold() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    record = {
        "folds": [
            {"fold": 0, "tta_oof_evaluation": {"mean_semantic_dice": 0.665}},
            {"fold": 1, "tta_oof_evaluation": {"mean_semantic_dice": 0.664}},
        ]
    }
    assert module.screen_semantic_gate(record, config=config) is True


def test_oof_metadata_is_written_before_test_images_are_loaded() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection" in source
