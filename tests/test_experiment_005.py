import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_005.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-005-hybrid-annotator.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_005", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_005_embeds_the_versioned_config() -> None:
    embedded_text = _embedded_config_yaml()
    versioned_text = CONFIG_PATH.read_text(encoding="utf-8").strip()

    assert embedded_text == versioned_text
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-005"


def test_experiment_005_uses_only_competition_checkpoints() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["model"]["competition_trained_checkpoint_source"] == "experiment-003"
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_hybrid_screen_is_fixed_before_confirmation() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["screen"]["folds"] == [0, 1]
    assert config["screen"]["reuse_selected_screen_models"] is True
    assert config["training"]["target_strategy"] == "hybrid-consensus-random-annotator"
    assert len(config["screen"]["weight_pairs"]) == 2
    for weights in config["screen"]["weight_pairs"]:
        assert weights["consensus"] + weights["annotator"] == 1.0


def test_annotator_sampling_is_deterministic_and_epoch_seeded() -> None:
    module = _load_kernel_module()

    first = [module.deterministic_annotator_index(17, epoch, 5, 3) for epoch in range(12)]
    second = [module.deterministic_annotator_index(17, epoch, 5, 3) for epoch in range(12)]

    assert first == second
    assert len(set(first)) > 1
    assert all(0 <= value < 3 for value in first)


def test_hybrid_loss_contains_both_declared_targets() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "hybrid_loss"
    )
    source = ast.unparse(function)

    assert "weights['consensus']" in source
    assert "weights['annotator']" in source
    assert "consensus_loss" in source
    assert "annotator_loss" in source
