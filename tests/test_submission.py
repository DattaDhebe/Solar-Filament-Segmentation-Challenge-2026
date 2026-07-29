import csv
from pathlib import Path

import numpy as np
import pytest

from solar_filament_segmentation.submission import (
    SubmissionRow,
    rows_for_image,
    validate_submission_rows,
    write_submission,
)


def test_rows_and_csv_use_required_columns(tmp_path: Path) -> None:
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 1:3] = 1
    second = np.zeros((8, 8), dtype=np.uint8)
    second[4:7, 4:7] = 1
    rows = rows_for_image("20150125172714Mh", [first, second])

    output_path = write_submission(
        rows,
        tmp_path / "submission.csv",
        known_image_ids={"20150125172714Mh"},
        height=8,
        width=8,
    )

    with output_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert list(records[0]) == ["filament_id", "segmentation_rle"]
    assert [record["filament_id"] for record in records] == [
        "20150125172714Mh_1",
        "20150125172714Mh_2",
    ]


def test_submission_rejects_duplicate_ids() -> None:
    rows = [
        SubmissionRow("image_1", "abc"),
        SubmissionRow("image_1", "def"),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        validate_submission_rows(rows, decode_masks=False)


def test_submission_rejects_unknown_image_id() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        validate_submission_rows(
            [SubmissionRow("unknown_1", "abc")],
            known_image_ids={"known"},
            decode_masks=False,
        )


def test_submission_rejects_empty_instance_mask() -> None:
    empty = np.zeros((8, 8), dtype=np.uint8)
    rows = rows_for_image("image", [empty])

    with pytest.raises(ValueError, match="Empty"):
        validate_submission_rows(rows, height=8, width=8)


def test_submission_rejects_duplicate_instance_masks() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    rows = rows_for_image("image", [mask, mask])

    with pytest.raises(ValueError, match="Duplicate instance mask"):
        validate_submission_rows(rows, height=8, width=8)
