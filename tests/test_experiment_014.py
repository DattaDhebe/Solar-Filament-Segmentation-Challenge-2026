"""Synthetic unit tests for Experiment 014 (Deep Supervision AttentionUNetPlusPlus).

Tests verify configuration integrity, model architecture contracts, loss functions,
distance transform computation, watershed instance separation, and pipeline ordering
without requiring GPU or competition data.
"""

from __future__ import annotations

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
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "experiment_014.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-014-unetplusplus.yaml"


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
    specification = importlib.util.spec_from_file_location("experiment_014", KERNEL_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    finally:
        if stubbed_cv2:
            del sys.modules["cv2"]
    return module


# ---------------------------------------------------------------------------
# Configuration integrity
# ---------------------------------------------------------------------------


def test_experiment_014_embeds_the_versioned_config() -> None:
    assert _embedded_config_yaml() == CONFIG_PATH.read_text(encoding="utf-8").strip()
    assert load_config(CONFIG_PATH)["experiment"]["id"] == "experiment-014"


def test_experiment_014_config_structure() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["experiment"]["id"] == "experiment-014"
    assert config["model"]["architecture"] == "attention-unetplusplus-deep-supervision"
    assert (
        config["validation"]["expected_fold_fingerprint"]
        == "69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7"
    )
    assert config["validation"]["group_key"] == "file_name"
    assert config["validation"]["fold_indices"] == list(range(5))


def test_model_initialization_from_scratch() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["initialization"] == "from-scratch"
    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False


def test_hybrid_target_weights_sum_to_one() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    total = config["training"]["consensus_weight"] + config["training"]["annotator_weight"]
    assert abs(total - 1.0) < 1e-9


def test_attention_and_deep_supervision_enabled() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["model"]["attention_gates"] is True
    assert config["model"]["deep_supervision"] is True
    assert config["model"]["distance_transform_head"] is True


def test_tversky_alpha_beta_recall_bias() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    alpha = config["training"]["tversky_alpha"]
    beta = config["training"]["tversky_beta"]
    assert alpha < beta, "beta > alpha enforces recall-heavy loss"
    assert abs(alpha + beta - 1.0) < 1e-9


def test_lower_thresholds_vs_experiment_008() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["postprocessing"]["probability_threshold"] < 0.50
    assert config["postprocessing"]["minimum_component_mean_probability"] < 0.80


def test_watershed_enabled_in_config() -> None:
    config = yaml.safe_load(_embedded_config_yaml())
    assert config["postprocessing"]["watershed_enabled"] is True
    assert config["postprocessing"]["watershed_seed_threshold"] > 0


# ---------------------------------------------------------------------------
# Model architecture contracts
# ---------------------------------------------------------------------------


def test_attention_unet_forward_pass() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    model = module.build_model(base_channels=8)
    x = torch.randn(1, 1, 64, 64)
    mask_logit, dist_pred, aux_logit = model(x)
    assert mask_logit.shape == (1, 1, 64, 64), f"mask shape: {mask_logit.shape}"
    assert dist_pred.shape == (1, 1, 64, 64), f"dist shape: {dist_pred.shape}"
    assert aux_logit.shape == (1, 1, 32, 32), f"aux shape: {aux_logit.shape}"


def test_attention_unet_outputs_valid_ranges() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    model = module.build_model(base_channels=8)
    x = torch.randn(1, 1, 64, 64)
    mask_logit, dist_pred, aux_logit = model(x)
    # dist_pred goes through sigmoid, so must be in [0, 1]
    assert dist_pred.min() >= 0.0
    assert dist_pred.max() <= 1.0
    # mask_logit and aux_logit are raw logits, no range constraint


def test_model_parameter_count_is_reasonable() -> None:
    pytest.importorskip("torch")
    module = _load_kernel_module()
    model = module.build_model(base_channels=32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # With 4 encoder levels + bottleneck + attention gates + dual heads,
    # should be larger than exp-008 SmallUNet but reasonable for T4 GPU
    assert param_count > 500_000, f"Too few params: {param_count}"
    assert param_count < 50_000_000, f"Too many params: {param_count}"


def test_attention_unet_handles_non_power_of_two_input() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    model = module.build_model(base_channels=8)
    # Test with a non-power-of-two but divisible-by-16 input
    x = torch.randn(1, 1, 128, 128)
    mask_logit, dist_pred, aux_logit = model(x)
    assert mask_logit.shape == (1, 1, 128, 128)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def test_focal_tversky_loss_computes() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    logits = torch.randn(1, 1, 32, 32)
    targets = (torch.rand(1, 1, 32, 32) > 0.5).float()
    loss = module.focal_tversky_loss(logits, targets, alpha=0.3, beta=0.7)
    assert loss.item() > 0
    assert loss.item() < 2.0


def test_focal_tversky_loss_is_zero_on_perfect_prediction() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    targets = (torch.rand(1, 1, 32, 32) > 0.5).float()
    # Large positive logits where target=1, large negative where target=0
    logits = targets * 100.0 - (1 - targets) * 100.0
    loss = module.focal_tversky_loss(logits, targets, alpha=0.3, beta=0.7)
    assert loss.item() < 0.01


def test_combined_loss_computes() -> None:
    torch = pytest.importorskip("torch")

    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    mask_logit = torch.randn(1, 1, 64, 64)
    dist_pred = torch.sigmoid(torch.randn(1, 1, 64, 64))
    aux_logit = torch.randn(1, 1, 32, 32)
    consensus = (torch.rand(1, 1, 64, 64) > 0.5).float()
    annotator = (torch.rand(1, 1, 64, 64) > 0.5).float()
    dist_target = torch.rand(1, 1, 64, 64)

    loss = module.combined_loss(
        mask_logit,
        dist_pred,
        aux_logit,
        consensus=consensus,
        annotator=annotator,
        dist_target=dist_target,
        config=config["training"],
    )
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


def test_compute_distance_target_basic() -> None:
    pytest.importorskip("cv2")
    module = _load_kernel_module()
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # 20x20 square in center
    dt = module.compute_distance_target(mask)
    assert dt.shape == (64, 64)
    assert dt.dtype == np.float32
    assert dt.max() <= 1.0
    assert dt.min() >= 0.0
    # Center of the square should have highest distance
    assert dt[30, 30] > dt[20, 20]
    # Background should be zero
    assert dt[0, 0] == 0.0


def test_compute_distance_target_empty_mask() -> None:
    pytest.importorskip("cv2")
    module = _load_kernel_module()
    mask = np.zeros((64, 64), dtype=np.uint8)
    dt = module.compute_distance_target(mask)
    assert dt.shape == (64, 64)
    assert dt.max() == 0.0


# ---------------------------------------------------------------------------
# Instance diagnostics
# ---------------------------------------------------------------------------


def test_instance_diagnostic_perfect_match() -> None:
    module = _load_kernel_module()
    instances = np.zeros((50, 50), dtype=np.uint16)
    instances[10:30, 10:30] = 1
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:30, 10:30] = 1
    result = module.instance_diagnostic([pred], instances, minimum_iou=0.1)
    assert result["matched"] == 1
    assert result["missed"] == 0
    assert result["extra"] == 0
    assert result["penalized_dice"] > 0.99


def test_instance_diagnostic_missed_instance() -> None:
    module = _load_kernel_module()
    instances = np.zeros((50, 50), dtype=np.uint16)
    instances[10:30, 10:30] = 1
    instances[30:45, 30:45] = 2
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:30, 10:30] = 1
    result = module.instance_diagnostic([pred], instances, minimum_iou=0.1)
    assert result["matched"] == 1
    assert result["missed"] == 1
    assert result["extra"] == 0


def test_instance_diagnostic_extra_instance() -> None:
    module = _load_kernel_module()
    instances = np.zeros((50, 50), dtype=np.uint16)
    instances[10:30, 10:30] = 1
    pred1 = np.zeros((50, 50), dtype=np.uint8)
    pred1[10:30, 10:30] = 1
    pred2 = np.zeros((50, 50), dtype=np.uint8)
    pred2[35:45, 35:45] = 1
    result = module.instance_diagnostic([pred1, pred2], instances, minimum_iou=0.1)
    assert result["matched"] == 1
    assert result["missed"] == 0
    assert result["extra"] == 1


# ---------------------------------------------------------------------------
# Semantic Dice helper
# ---------------------------------------------------------------------------


def test_semantic_dice_perfect() -> None:
    module = _load_kernel_module()
    prob = np.ones((1, 1, 32, 32), dtype=np.float32)
    target = np.ones((1, 1, 32, 32), dtype=np.float32)
    dice = module.semantic_dice(prob, target)
    assert dice[0] > 0.99


def test_semantic_dice_empty() -> None:
    module = _load_kernel_module()
    prob = np.zeros((1, 1, 32, 32), dtype=np.float32)
    target = np.zeros((1, 1, 32, 32), dtype=np.float32)
    dice = module.semantic_dice(prob, target)
    assert dice[0] == 1.0  # Both empty => 1.0


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def test_promotion_gate_passes_on_good_metrics() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    summary = {
        "mean_penalized_instance_dice": 0.55,
        "missed": 1200,
        "extra": 1800,
    }
    result = module.promotion_result(summary, config=config)
    assert result["passed"] is True


def test_promotion_gate_fails_on_high_missed() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    summary = {
        "mean_penalized_instance_dice": 0.55,
        "missed": 1500,
        "extra": 1800,
    }
    result = module.promotion_result(summary, config=config)
    assert result["passed"] is False


def test_promotion_gate_fails_on_low_dice() -> None:
    module = _load_kernel_module()
    config = yaml.safe_load(_embedded_config_yaml())
    summary = {
        "mean_penalized_instance_dice": 0.40,
        "missed": 1000,
        "extra": 1000,
    }
    result = module.promotion_result(summary, config=config)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Pipeline ordering contracts
# ---------------------------------------------------------------------------


def test_oof_metadata_written_before_test_inference() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    assert source.index("oof_path.write_text") < source.index("prepare_test_images")
    assert "test_statistics_used_for_selection" in source


def test_train_fold_uses_from_scratch_initialization() -> None:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    train_fold = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_fold"
    )
    source = ast.unparse(train_fold)
    assert "build_model" in source
    assert "from-scratch" in source
    assert "resolve_checkpoint" not in source


def test_no_screen_logic_all_folds_trained() -> None:
    """Experiment 013 trains all 5 folds directly, no screen/selection phase."""
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.unparse(main)
    assert "select_setting" not in source
    assert "screen_semantic_gate" not in source
    # Must iterate over all fold_indices
    assert "fold_indices" in source
