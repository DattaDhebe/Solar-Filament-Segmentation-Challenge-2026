import json
from pathlib import Path

from solar_filament_segmentation.data import (
    load_coco_annotations,
    physical_observation_key,
    summarize_coco,
)


def test_summary_counts_duplicate_annotator_views(tmp_path: Path) -> None:
    document = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "Left"},
            {"id": 2, "name": "Right"},
        ],
        "images": [
            {"id": "010101-observation", "file_name": "observation.jpeg"},
            {"id": "010102-observation", "file_name": "observation.jpeg"},
            {"id": "010101-second", "file_name": "second.jpeg"},
        ],
        "annotations": [{"id": "one"}, {"id": "two"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_coco_annotations(path)
    summary = summarize_coco(loaded)

    assert physical_observation_key(loaded["images"][0]) == "observation"
    assert summary.image_records == 3
    assert summary.physical_observations == 2
    assert summary.duplicate_annotation_sets == 1
    assert summary.annotation_records == 2
    assert summary.categories == ("Left", "Right")
