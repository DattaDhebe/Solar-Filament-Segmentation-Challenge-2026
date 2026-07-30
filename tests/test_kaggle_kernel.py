import ast
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPOSITORY_ROOT / "kaggle" / "first_submission.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-003-oof-tta-refinement.yaml"


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


def test_kaggle_kernel_embeds_the_versioned_experiment_config() -> None:
    embedded = yaml.safe_load(_embedded_config_yaml())
    versioned = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert embedded == versioned


def test_kaggle_kernel_does_not_enable_external_assets() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["model"]["pretrained_weights"] is False
    assert config["model"]["external_labeled_data"] is False


def test_overnight_kernel_covers_every_grouped_fold() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["validation"]["fold_indices"] == list(range(config["validation"]["n_splits"]))


def test_refinement_uses_only_competition_trained_checkpoints() -> None:
    config = yaml.safe_load(_embedded_config_yaml())

    assert config["experiment"]["mode"] == "checkpoint-finetuning-and-refinement"
    assert config["model"]["competition_trained_checkpoint_source"] == "experiment-002"
    assert config["training"]["target_strategy"] == "annotator-soft-consensus"
