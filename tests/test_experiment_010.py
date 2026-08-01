import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_010.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-010-postprocessing-calibration.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_010", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_010_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-010"


def test_calibration_grid_contains_all_declared_candidates_and_baseline() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    candidates = module.calibration_candidates(config)
    assert len(candidates) == 144
    assert config["calibration"]["baseline_candidate"] in candidates
    assert len({tuple(sorted(candidate.items())) for candidate in candidates}) == 144


def test_calibration_selection_requires_all_promotion_constraints() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    candidates = module.calibration_candidates(config)
    good = candidates[0]
    bad = candidates[1]
    results = [
        {
            "candidate": good,
            "evaluation": {
                "instance": {
                    "mean_penalized_instance_dice": 0.51,
                    "mean_matched_instance_dice": 0.73,
                    "missed": 1660,
                    "extra": 1930,
                }
            },
        },
        {
            "candidate": bad,
            "evaluation": {
                "instance": {
                    "mean_penalized_instance_dice": 0.52,
                    "mean_matched_instance_dice": 0.74,
                    "missed": 1660,
                    "extra": 2000,
                }
            },
        },
    ]
    selected = module.select_calibration_candidate(results, config=config)
    assert selected is results[0]
    assert selected["feasible"] is True
    assert results[1]["feasible"] is False


def test_calibration_uses_only_experiment_008_checkpoints() -> None:
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
