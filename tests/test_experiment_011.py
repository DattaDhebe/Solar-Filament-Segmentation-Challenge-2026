import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_011.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-011-multiscale-tta.yaml"


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
    return str(ast.literal_eval(strip_call.func.value)).strip()


def _load_kernel_module():
    stubbed_cv2 = "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None
    if stubbed_cv2:
        sys.modules["cv2"] = types.ModuleType("cv2")
    specification = importlib.util.spec_from_file_location("experiment_011", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_011_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-011"


def test_tta_screen_has_baseline_and_two_multiscale_variants() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    variants = config["inference"]["variants"]
    assert [variant["id"] for variant in variants] == [
        "baseline-flips",
        "multiscale-075-100-125",
        "multiscale-0875-100-1125",
    ]
    assert variants[0]["scales"] == [1.0]
    assert all(scale > 0 for variant in variants for scale in variant["scales"])


def test_multiscale_tta_resizes_to_divisible_by_eight_scales() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "predict_with_multiscale_tta"
    )
    source = ast.unparse(function)
    assert "round(base_height * scale / 8.0)" in source
    assert "functional.interpolate" in source
    assert "torch.flip" in source


def test_tta_uses_only_experiment_008_checkpoints() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["checkpoint_experiment"] == "experiment-008"
    assert config["model"]["initialization"] == "experiment-008-selected"
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False


def test_oof_metadata_is_written_before_test_images_are_loaded() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection" in source


def test_variant_selection_falls_back_to_baseline_when_gate_fails() -> None:
    module = _load_kernel_module()
    results = [
        {
            "variant": {"id": "baseline-flips"},
            "feasible": False,
            "instance": {"mean_penalized_instance_dice": 0.495},
        },
        {
            "variant": {"id": "multiscale"},
            "feasible": False,
            "instance": {"mean_penalized_instance_dice": 0.501},
        },
    ]
    assert module.select_variant_result(results)["variant"]["id"] == "baseline-flips"
