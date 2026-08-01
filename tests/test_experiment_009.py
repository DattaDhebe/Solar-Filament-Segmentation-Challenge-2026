import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_009.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-009-low-lr-continuation.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_009", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_009_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-009"


def test_continuation_uses_only_experiment_008_selected_checkpoints() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["initialization"] == "experiment-008-selected"
    assert config["model"]["checkpoint_experiment"] == "experiment-008"
    assert config["model"]["input_size"] == 1024
    assert config["model"]["base_channels"] == 32
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_continuation_learning_rate_is_lower_than_experiment_008() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["training"]["learning_rate"] == 0.00005
    assert config["training"]["epochs"] == 14
    assert config["training"]["gradient_accumulation_steps"] == 2


def test_train_fold_loads_declared_checkpoint_source() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    train_fold = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_fold"
    )
    source = ast.unparse(train_fold)
    assert "resolve_checkpoint(config, fold_index)" in source
    assert "torch.load(checkpoint" in source
    assert "config['model']['initialization']" in source


def test_oof_metadata_is_written_before_test_images_are_loaded() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection" in source
