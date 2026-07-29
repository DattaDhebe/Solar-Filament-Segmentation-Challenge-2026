import numpy as np
import pytest

from solar_filament_segmentation.metrics import binary_dice, binary_iou


def test_binary_metrics_for_known_overlap() -> None:
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    assert binary_dice(prediction, target) == pytest.approx(0.5)
    assert binary_iou(prediction, target) == pytest.approx(1 / 3)


def test_binary_metrics_treat_two_empty_masks_as_perfect() -> None:
    empty = np.zeros((3, 4), dtype=np.uint8)

    assert binary_dice(empty, empty) == 1.0
    assert binary_iou(empty, empty) == 1.0


def test_binary_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        binary_dice(np.zeros((2, 2)), np.zeros((3, 2)))
