from pathlib import Path

import pytest
import yaml

from solar_filament_segmentation.config import ConfigError, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment-000-workspace-smoke.yaml"


def test_workspace_config_is_valid() -> None:
    config = load_config(CONFIG_PATH)

    assert config["competition"]["image_height"] == 2048
    assert config["validation"]["group_key"] == "file_name"
    assert config["model"]["external_labeled_data"] is False


def test_config_rejects_annotation_id_grouping(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["validation"]["group_key"] = "id"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="file_name"):
        load_config(invalid_path)
