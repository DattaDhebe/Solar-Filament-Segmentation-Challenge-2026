import numpy as np

from solar_filament_segmentation.rle import (
    decode_rle_counts,
    encode_binary_mask,
    polygons_to_mask,
)


def test_compressed_rle_round_trip() -> None:
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[2:6, 3:8] = 1

    counts = encode_binary_mask(mask)
    decoded = decode_rle_counts(counts, height=8, width=9)

    assert isinstance(counts, str)
    assert '"' not in counts
    assert "'" not in counts
    np.testing.assert_array_equal(decoded, mask)


def test_polygon_rasterization() -> None:
    polygon = [[1, 1, 5, 1, 5, 5, 1, 5, 1, 1]]

    mask = polygons_to_mask(polygon, height=8, width=8)

    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_competition_size_sparse_mask_round_trip() -> None:
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[101:109, 1501:1517] = 1

    counts = encode_binary_mask(mask)
    decoded = decode_rle_counts(counts)

    np.testing.assert_array_equal(decoded, mask)
