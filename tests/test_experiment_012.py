import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import yaml

from solar_filament_segmentation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_012.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-012-patch2048-clahe.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_012", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


def test_experiment_012_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-012"


def test_clahe_preprocessing_helper() -> None:
    module = _load_kernel_module()
    sample_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    enhanced = module.apply_clahe_preprocessing(sample_img, clip_limit=2.0)
    assert enhanced.shape == (512, 512)
    assert enhanced.dtype == np.uint8


def test_patch_cropping_helper() -> None:
    module = _load_kernel_module()
    sample_img = np.zeros((2048, 2048), dtype=np.uint8)
    sample_mask = np.zeros((2048, 2048), dtype=np.uint8)
    img_patch, mask_patch = module.crop_patch(sample_img, sample_mask, crop_size=1024, seed=42)
    assert img_patch.shape == (1024, 1024)
    assert mask_patch.shape == (1024, 1024)


def test_model_initialization_from_scratch() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["initialization"] == "from-scratch"
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False
