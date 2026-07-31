"""Kaggle GPU script for experiment-004 component ranking and fragment linking.

The script is self-contained because Kaggle executes the configured code file.
Keep ``EMBEDDED_CONFIG_YAML`` synchronized with
``configs/experiment-004-component-quality.yaml``; a synthetic test enforces it.

Experiment 004 never trains on or selects from test observations. It derives
component-quality targets only from grouped out-of-fold predictions, cross-fits
the quality ranker by the immutable validation fold, freezes the selected OOF
settings, and only then performs final test inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

EMBEDDED_CONFIG_YAML = """
experiment:
  id: experiment-004
  name: cross-fitted-component-quality
  seed: 20260730
  output_root: outputs/experiments
  mode: checkpoint-inference-and-cross-fitted-ranking

competition:
  slug: filament-segmentation-2026
  image_height: 2048
  image_width: 2048
  grayscale: true

data:
  local_root: data/raw/MAGFiLO_1.0_Kaggle_2026
  kaggle_root: /kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
  train_images: train/train_images
  train_annotations: train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test_images: test/test_images
  checkpoint_root: /kaggle/input/solar-filament-oof-tta-refinement

validation:
  strategy: grouped-k-fold
  group_key: file_name
  n_splits: 5
  fold_indices:
    - 0
    - 1
    - 2
    - 3
    - 4
  shuffle: true
  seed: 20260729
  expected_fold_fingerprint: 69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7
  component_ranker_cross_fit_by: validation_fold

model:
  task: instance-segmentation
  architecture: small-unet
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 24
  pretrained_weights: false
  external_labeled_data: false
  competition_trained_checkpoint_source: experiment-003
  checkpoint_training_enabled: false

inference:
  batch_size: 2
  test_time_augmentations:
    - identity
    - horizontal-flip
    - vertical-flip
    - horizontal-vertical-flip

candidate_pool:
  probability_threshold: 0.4
  closing_kernel: 5
  closing_iterations: 1
  disk_erosion_pixels: 8
  minimum_component_area_at_model_resolution: 16
  maximum_component_area_at_model_resolution: 40000
  connectivity: 8

component_quality:
  model: histogram-gradient-boosting-regressor
  seed: 20260730
  max_depth: 3
  max_iter: 150
  learning_rate: 0.05
  l2_regularization: 0.1
  quality_cutoffs:
    - 0.1
    - 0.15
    - 0.2
    - 0.25
    - 0.3
  cross_fit_by: validation_fold
  target: mean-maximum-iou-across-annotators

fragment_linking:
  enabled_candidates:
    - false
    - true
  maximum_endpoint_distances:
    - 8
    - 16
    - 24
  minimum_tangent_cosines:
    - 0.8
    - 0.9
  minimum_gap_probabilities:
    - 0.3
    - 0.4
    - 0.5
  line_thickness: 3

selection:
  matching_min_iou: 0.1
  primary_metric: mean-penalized-instance-dice
  tie_breakers:
    - fewer-extra-instances
    - fewer-missed-instances
    - linking-disabled
  baseline:
    experiment_id: experiment-003
    mean_penalized_instance_dice: 0.490129
    missed_instances: 1693
    extra_instances: 2103
  promotion_gate:
    minimum_mean_penalized_instance_dice: 0.505
    minimum_extra_instance_reduction_fraction: 0.15
    maximum_missed_instance_increase_fraction: 0.05

metric:
  primary: mean-matched-instance-dice
  diagnostics:
    - semantic-dice-distribution
    - matched-instance-dice
    - matched-instance-iou
    - missed-and-extra-instances
    - one-to-many-relations
    - many-to-one-relations
    - component-quality-calibration
    - per-image-instance-dice-distribution

submission:
  columns:
    - filament_id
    - segmentation_rle
  rle_format: coco-compressed-counts
  mask_height: 2048
  mask_width: 2048
  output_path: /kaggle/working/experiment-004-submission.csv
""".strip()

FEATURE_NAMES = (
    "area",
    "perimeter",
    "mean_probability",
    "maximum_probability",
    "probability_standard_deviation",
    "probability_q25",
    "probability_q50",
    "probability_q75",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "bounding_box_aspect_ratio",
    "skeleton_length",
    "estimated_width",
    "distance_from_disk_center",
    "normalized_radial_position",
    "distance_to_disk_boundary",
    "nearest_component_distance",
    "nearest_component_tangent_alignment",
)


def seed_everything(seed: int) -> None:
    """Make model execution and the component ranker reproducible."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_data_root(config: dict) -> Path:
    """Resolve only the official competition data root."""
    configured = Path(config["data"]["kaggle_root"])
    annotation_relative = Path(config["data"]["train_annotations"])
    if (configured / annotation_relative).is_file():
        return configured

    candidates = sorted(
        Path("/kaggle/input").glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    )
    valid = [
        path.parents[1]
        for path in candidates
        if path.parent.name == "train" and (path.parents[1] / "test" / "test_images").is_dir()
    ]
    if len(valid) != 1:
        raise FileNotFoundError(
            "Could not uniquely resolve the official MAGFiLO competition root; "
            f"configured={configured}, candidates={valid}."
        )
    return valid[0]


def resolve_checkpoint(config: dict, fold_index: int) -> Path:
    """Resolve only an experiment-003 checkpoint trained from competition data."""
    file_name = f"experiment-003-fold-{fold_index}.pt"
    configured = Path(config["data"]["checkpoint_root"]) / file_name
    if configured.is_file():
        return configured

    candidates = sorted(Path("/kaggle/input").glob(f"**/{file_name}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {file_name}; "
            f"configured={configured}, candidates={candidates}."
        )
    return candidates[0]


def load_observations(annotation_path: Path) -> tuple[list[dict], int, int]:
    """Group official annotation sets by the physical JPEG file name."""
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    polygons_by_image: dict[object, list[list[float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("This pipeline supports the supplied polygon annotations only.")
        polygons_by_image[annotation["image_id"]].extend(segmentation)

    grouped: dict[str, dict] = {}
    for image_record in document["images"]:
        file_name = str(image_record["file_name"])
        observation_key = Path(file_name).stem
        observation = grouped.setdefault(
            observation_key,
            {
                "observation_key": observation_key,
                "file_name": file_name,
                "annotation_sets": [],
            },
        )
        if observation["file_name"] != file_name:
            raise ValueError(f"Observation key collision for {observation_key}.")
        observation["annotation_sets"].append(
            {
                "image_record_id": image_record["id"],
                "polygons": polygons_by_image.get(image_record["id"], []),
            }
        )

    observations = [grouped[key] for key in sorted(grouped)]
    return observations, len(document["images"]), len(document["annotations"])


def assign_grouped_folds(
    observations: list[dict],
    *,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, int], str]:
    """Assign each physical observation to one immutable deterministic fold."""
    observation_keys = sorted(str(item["observation_key"]) for item in observations)
    if n_splits < 2 or n_splits > len(observation_keys):
        raise ValueError("n_splits must be between 2 and the physical-observation count.")
    random.Random(seed).shuffle(observation_keys)
    assignments = {key: index % n_splits for index, key in enumerate(observation_keys)}
    canonical = "\n".join(f"{key},{assignments[key]}" for key in sorted(assignments))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return assignments, fingerprint


def rasterize_instance_labels(
    polygons: list[list[float]],
    *,
    original_size: int,
    model_size: int,
) -> np.ndarray:
    """Rasterize one official annotator record into integer instance labels."""
    if not polygons:
        return np.zeros((model_size, model_size), dtype=np.uint16)
    scale = model_size / original_size
    scaled = [[coordinate * scale for coordinate in polygon] for polygon in polygons]
    rles = mask_utils.frPyObjects(scaled, model_size, model_size)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, np.newaxis]
    labels = np.zeros((model_size, model_size), dtype=np.uint16)
    for index in range(decoded.shape[2]):
        labels[decoded[:, :, index] != 0] = index + 1
    return labels


def load_grayscale_image(image_path: Path, *, model_size: int) -> np.ndarray:
    """Load one H-alpha JPEG as a resized grayscale array."""
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(
            (model_size, model_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(resized, dtype=np.uint8)


def build_model(base_channels: int):
    """Reconstruct the compact U-Net used by experiment 003."""
    import torch
    from torch import nn

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)

    class SmallUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = base_channels
            self.encoder1 = ConvBlock(1, channels)
            self.encoder2 = ConvBlock(channels, channels * 2)
            self.encoder3 = ConvBlock(channels * 2, channels * 4)
            self.bottleneck = ConvBlock(channels * 4, channels * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
            self.decoder3 = ConvBlock(channels * 8, channels * 4)
            self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
            self.decoder2 = ConvBlock(channels * 4, channels * 2)
            self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
            self.decoder1 = ConvBlock(channels * 2, channels)
            self.head = nn.Conv2d(channels, 1, 1)

        def forward(self, inputs):
            encoded1 = self.encoder1(inputs)
            encoded2 = self.encoder2(self.pool(encoded1))
            encoded3 = self.encoder3(self.pool(encoded2))
            bottleneck = self.bottleneck(self.pool(encoded3))
            decoded3 = self.decoder3(torch.cat((self.up3(bottleneck), encoded3), dim=1))
            decoded2 = self.decoder2(torch.cat((self.up2(decoded3), encoded2), dim=1))
            decoded1 = self.decoder1(torch.cat((self.up1(decoded2), encoded1), dim=1))
            return self.head(decoded1)

    return SmallUNet()


def predict_with_tta(model, inputs, *, augmentations: list[str]):
    """Average aligned probabilities across deterministic geometric transforms."""
    import torch

    probabilities = []
    for augmentation in augmentations:
        if augmentation == "identity":
            transformed = inputs
            inverse_dimensions: tuple[int, ...] = ()
        elif augmentation == "horizontal-flip":
            inverse_dimensions = (-1,)
            transformed = torch.flip(inputs, inverse_dimensions)
        elif augmentation == "vertical-flip":
            inverse_dimensions = (-2,)
            transformed = torch.flip(inputs, inverse_dimensions)
        elif augmentation == "horizontal-vertical-flip":
            inverse_dimensions = (-2, -1)
            transformed = torch.flip(inputs, inverse_dimensions)
        else:
            raise ValueError(f"Unsupported test-time augmentation: {augmentation}")
        probability = torch.sigmoid(model(transformed))
        if inverse_dimensions:
            probability = torch.flip(probability, inverse_dimensions)
        probabilities.append(probability)
    return torch.stack(probabilities).mean(dim=0)


def predict_image(model, image: np.ndarray, *, device, augmentations: list[str]) -> np.ndarray:
    """Predict one model-resolution probability map."""
    import torch

    tensor = torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    model.eval()
    with torch.no_grad():
        probability = predict_with_tta(
            model,
            tensor,
            augmentations=augmentations,
        )[0, 0]
    return probability.cpu().numpy().astype(np.float16)


def solar_disk_geometry(
    image: np.ndarray,
    *,
    erosion_pixels: int,
) -> tuple[np.ndarray, tuple[float, float], float, np.ndarray]:
    """Estimate the eroded solar disk, center, radius, and boundary distance."""
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    label_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        thresholded,
        connectivity=8,
    )
    if label_count <= 1:
        disk = np.ones_like(image, dtype=np.uint8)
        center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
        radius = min(image.shape) / 2.0
    else:
        largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        disk = (labels == largest).astype(np.uint8)
        center = (float(centroids[largest][0]), float(centroids[largest][1]))
        radius = math.sqrt(float(statistics[largest, cv2.CC_STAT_AREA]) / math.pi)
    if erosion_pixels > 0:
        kernel_size = erosion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        disk = cv2.erode(disk, kernel, iterations=1)
    boundary_distance = cv2.distanceTransform(disk, cv2.DIST_L2, 5)
    return disk != 0, center, max(radius, 1.0), boundary_distance


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return a deterministic binary skeleton using OpenCV morphology."""
    working = (mask != 0).astype(np.uint8)
    skeleton = np.zeros_like(working)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(working):
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= working & (opened == 0)
        working = eroded
    return skeleton


def principal_geometry(
    mask: np.ndarray,
) -> tuple[float, float, float, np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    """Measure principal axes, eccentricity, tangent, and representative endpoints."""
    y_coordinates, x_coordinates = np.nonzero(mask)
    coordinates = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
    if len(coordinates) < 2:
        point = (int(x_coordinates[0]), int(y_coordinates[0]))
        return 1.0, 1.0, 0.0, np.array([1.0, 0.0]), (point, point)

    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_variance = max(float(eigenvalues[order[0]]), 0.0)
    minor_variance = max(float(eigenvalues[order[1]]), 0.0)
    tangent = eigenvectors[:, order[0]]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    centered = coordinates - coordinates.mean(axis=0)
    projections = centered @ tangent
    endpoints = (
        tuple(coordinates[int(np.argmin(projections))].astype(int)),
        tuple(coordinates[int(np.argmax(projections))].astype(int)),
    )
    major_axis = max(4.0 * math.sqrt(major_variance), 1.0)
    minor_axis = max(4.0 * math.sqrt(minor_variance), 1.0)
    eccentricity = math.sqrt(max(0.0, 1.0 - (minor_variance / max(major_variance, 1e-8))))
    return major_axis, minor_axis, eccentricity, tangent, endpoints


def extract_components(
    probability: np.ndarray,
    image: np.ndarray,
    *,
    config: dict,
) -> tuple[list[dict], np.ndarray]:
    """Create a permissive component pool and deterministic shape features."""
    disk, disk_center, disk_radius, boundary_distance = solar_disk_geometry(
        image,
        erosion_pixels=config["disk_erosion_pixels"],
    )
    foreground = (probability >= config["probability_threshold"]) & disk
    kernel_size = config["closing_kernel"]
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config["closing_iterations"],
        )
        foreground = (foreground != 0) & disk

    label_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=config["connectivity"],
    )
    components = []
    for label in range(1, label_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if not (
            config["minimum_component_area_at_model_resolution"]
            <= area
            <= config["maximum_component_area_at_model_resolution"]
        ):
            continue
        component = labels == label
        values = probability[component].astype(np.float32)
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        hull_area = 0.0
        for contour in contours:
            if len(contour) >= 3:
                hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = min(1.0, area / max(hull_area, float(area), 1.0))
        major, minor, eccentricity, tangent, fallback_endpoints = principal_geometry(component)

        left = int(statistics[label, cv2.CC_STAT_LEFT])
        top = int(statistics[label, cv2.CC_STAT_TOP])
        width = int(statistics[label, cv2.CC_STAT_WIDTH])
        height = int(statistics[label, cv2.CC_STAT_HEIGHT])
        crop = component[top : top + height, left : left + width]
        skeleton_crop = morphological_skeleton(crop)
        skeleton_length = int(np.count_nonzero(skeleton_crop))
        neighbor_count = cv2.filter2D(
            skeleton_crop,
            cv2.CV_16U,
            np.ones((3, 3), dtype=np.uint8),
        )
        endpoint_y, endpoint_x = np.nonzero((skeleton_crop != 0) & (neighbor_count == 2))
        if len(endpoint_x) >= 2:
            endpoint_coordinates = np.column_stack((endpoint_x + left, endpoint_y + top)).astype(
                np.float64
            )
            projections = endpoint_coordinates @ tangent
            endpoints = (
                tuple(endpoint_coordinates[int(np.argmin(projections))].astype(int)),
                tuple(endpoint_coordinates[int(np.argmax(projections))].astype(int)),
            )
        else:
            endpoints = fallback_endpoints

        centroid_x, centroid_y = centroids[label]
        radial_distance = math.hypot(centroid_x - disk_center[0], centroid_y - disk_center[1])
        boundary_value = float(boundary_distance[int(round(centroid_y)), int(round(centroid_x))])
        quantiles = np.quantile(values, [0.25, 0.5, 0.75])
        base_features = [
            float(area),
            perimeter,
            float(values.mean()),
            float(values.max()),
            float(values.std()),
            float(quantiles[0]),
            float(quantiles[1]),
            float(quantiles[2]),
            major,
            minor,
            eccentricity,
            solidity,
            width / max(height, 1),
            float(skeleton_length),
            area / max(skeleton_length, 1),
            radial_distance,
            radial_distance / disk_radius,
            boundary_value,
        ]
        components.append(
            {
                "mask": component.astype(np.uint8),
                "pixel_indices": np.flatnonzero(component),
                "centroid": np.array([centroid_x, centroid_y], dtype=np.float64),
                "tangent": tangent,
                "endpoints": endpoints,
                "base_features": base_features,
            }
        )

    for index, component in enumerate(components):
        nearest_distance = float(max(image.shape))
        nearest_alignment = 0.0
        for other_index, other in enumerate(components):
            if other_index == index:
                continue
            distance = float(np.linalg.norm(component["centroid"] - other["centroid"]))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_alignment = abs(float(np.dot(component["tangent"], other["tangent"])))
        features = np.asarray(
            [*component["base_features"], nearest_distance, nearest_alignment],
            dtype=np.float64,
        )
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            raise ValueError("Component feature extraction produced invalid values.")
        component["features"] = features
        del component["base_features"]
    return components, disk


def maximum_iou(pixel_indices: np.ndarray, target_labels: np.ndarray) -> float:
    """Return maximum IoU between one candidate and any official target instance."""
    component_area = len(pixel_indices)
    flat_targets = target_labels.ravel()
    target_counts = np.bincount(flat_targets)
    overlapping = flat_targets[pixel_indices]
    target_ids, intersections = np.unique(overlapping, return_counts=True)
    best = 0.0
    for target_id, intersection in zip(target_ids, intersections, strict=True):
        target_id = int(target_id)
        if target_id == 0:
            continue
        target_area = int(target_counts[target_id])
        union = component_area + target_area - int(intersection)
        best = max(best, int(intersection) / max(union, 1))
    return best


def component_targets(components: list[dict], target_sets: list[np.ndarray]) -> np.ndarray:
    """Average maximum IoU across all annotators of one physical observation."""
    return np.asarray(
        [
            float(
                np.mean([maximum_iou(component["pixel_indices"], labels) for labels in target_sets])
            )
            for component in components
        ],
        dtype=np.float64,
    )


def make_ranker(config: dict):
    """Create the deterministic low-capacity component-quality regressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=config["max_depth"],
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        l2_regularization=config["l2_regularization"],
        random_state=config["seed"],
    )


def cross_fitted_scores(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    observation_keys: np.ndarray,
    *,
    fold_indices: list[int],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Fit on four OOF component folds and predict the untouched fifth fold."""
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    records = []
    for validation_fold in fold_indices:
        training = folds != validation_fold
        validation = folds == validation_fold
        if set(observation_keys[training]) & set(observation_keys[validation]):
            raise RuntimeError("Component-ranker cross-fitting leaked a file_name group.")
        ranker = make_ranker(config)
        ranker.fit(features[training], targets[training])
        scores[validation] = np.clip(ranker.predict(features[validation]), 0.0, 1.0)
        records.append(
            {
                "validation_fold": validation_fold,
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "training_observations": len(set(observation_keys[training])),
                "validation_observations": len(set(observation_keys[validation])),
                "target_mean": float(targets[validation].mean()),
                "score_mean": float(scores[validation].mean()),
                "mean_absolute_error": float(
                    np.mean(np.abs(scores[validation] - targets[validation]))
                ),
            }
        )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Every component must receive exactly one cross-fitted score.")
    return scores, records


def instance_diagnostic(
    predicted_instances: list[np.ndarray],
    target_labels: np.ndarray,
    *,
    minimum_iou: float,
) -> dict[str, float | int]:
    """Greedily match instances and penalize missed and extra predictions."""
    target_ids = np.unique(target_labels)
    target_ids = target_ids[target_ids != 0]
    target_areas = {
        int(target_id): int(np.count_nonzero(target_labels == target_id))
        for target_id in target_ids
    }
    pairs = []
    overlap_by_prediction = []
    overlap_by_target = {int(target_id): 0 for target_id in target_ids}
    for prediction_index, prediction in enumerate(predicted_instances):
        prediction_area = int(prediction.sum())
        overlapping_ids, intersections = np.unique(
            target_labels[prediction != 0],
            return_counts=True,
        )
        overlapping_targets = 0
        for target_id, intersection in zip(overlapping_ids, intersections, strict=True):
            target_id = int(target_id)
            if target_id == 0:
                continue
            intersection = int(intersection)
            union = prediction_area + target_areas[target_id] - intersection
            iou = intersection / union
            dice = (2.0 * intersection) / (prediction_area + target_areas[target_id])
            if iou >= minimum_iou:
                pairs.append((dice, iou, prediction_index, target_id))
                overlapping_targets += 1
                overlap_by_target[target_id] += 1
        overlap_by_prediction.append(overlapping_targets)

    matched_predictions = set()
    matched_targets = set()
    matched_dice = []
    matched_iou = []
    for dice, iou, prediction_index, target_id in sorted(pairs, reverse=True):
        if prediction_index in matched_predictions or target_id in matched_targets:
            continue
        matched_predictions.add(prediction_index)
        matched_targets.add(target_id)
        matched_dice.append(dice)
        matched_iou.append(iou)
    denominator = max(len(predicted_instances), len(target_ids), 1)
    return {
        "penalized_dice": float(sum(matched_dice) / denominator),
        "matched_dice_sum": float(sum(matched_dice)),
        "matched_iou_sum": float(sum(matched_iou)),
        "matched": len(matched_dice),
        "missed": len(target_ids) - len(matched_targets),
        "extra": len(predicted_instances) - len(matched_predictions),
        "one_to_many": sum(count > 1 for count in overlap_by_target.values()),
        "many_to_one": sum(count > 1 for count in overlap_by_prediction),
    }


def sparse_instance_diagnostic(
    predicted_instances: list[dict],
    target_labels: np.ndarray,
    *,
    minimum_iou: float,
) -> dict[str, float | int]:
    """Match instances using cached component pixels instead of full-frame scans."""
    flat_targets = target_labels.ravel()
    target_counts = np.bincount(flat_targets)
    target_ids = np.flatnonzero(target_counts)
    target_ids = target_ids[target_ids != 0]
    pairs = []
    overlap_by_prediction = []
    overlap_by_target = {int(target_id): 0 for target_id in target_ids}
    for prediction_index, prediction in enumerate(predicted_instances):
        pixel_indices = prediction["pixel_indices"]
        prediction_area = len(pixel_indices)
        overlapping_ids, intersections = np.unique(
            flat_targets[pixel_indices],
            return_counts=True,
        )
        overlapping_targets = 0
        for target_id, intersection in zip(overlapping_ids, intersections, strict=True):
            target_id = int(target_id)
            if target_id == 0:
                continue
            intersection = int(intersection)
            target_area = int(target_counts[target_id])
            union = prediction_area + target_area - intersection
            iou = intersection / union
            dice = (2.0 * intersection) / (prediction_area + target_area)
            if iou >= minimum_iou:
                pairs.append((dice, iou, prediction_index, target_id))
                overlapping_targets += 1
                overlap_by_target[target_id] += 1
        overlap_by_prediction.append(overlapping_targets)

    matched_predictions = set()
    matched_targets = set()
    matched_dice = []
    matched_iou = []
    for dice, iou, prediction_index, target_id in sorted(pairs, reverse=True):
        if prediction_index in matched_predictions or target_id in matched_targets:
            continue
        matched_predictions.add(prediction_index)
        matched_targets.add(target_id)
        matched_dice.append(dice)
        matched_iou.append(iou)
    denominator = max(len(predicted_instances), len(target_ids), 1)
    return {
        "penalized_dice": float(sum(matched_dice) / denominator),
        "matched_dice_sum": float(sum(matched_dice)),
        "matched_iou_sum": float(sum(matched_iou)),
        "matched": len(matched_dice),
        "missed": len(target_ids) - len(matched_targets),
        "extra": len(predicted_instances) - len(matched_predictions),
        "one_to_many": sum(count > 1 for count in overlap_by_target.values()),
        "many_to_one": sum(count > 1 for count in overlap_by_prediction),
    }


def linking_candidates(config: dict) -> list[dict]:
    """Expand the frozen OOF-only component cutoff and fragment-linking grid."""
    settings = []
    for cutoff in config["component_quality"]["quality_cutoffs"]:
        if False in config["fragment_linking"]["enabled_candidates"]:
            settings.append(
                {
                    "quality_cutoff": cutoff,
                    "linking_enabled": False,
                    "maximum_endpoint_distance": 0,
                    "minimum_tangent_cosine": 1.0,
                    "minimum_gap_probability": 1.0,
                }
            )
        if True in config["fragment_linking"]["enabled_candidates"]:
            for distance in config["fragment_linking"]["maximum_endpoint_distances"]:
                for cosine in config["fragment_linking"]["minimum_tangent_cosines"]:
                    for gap_probability in config["fragment_linking"]["minimum_gap_probabilities"]:
                        settings.append(
                            {
                                "quality_cutoff": cutoff,
                                "linking_enabled": True,
                                "maximum_endpoint_distance": distance,
                                "minimum_tangent_cosine": cosine,
                                "minimum_gap_probability": gap_probability,
                            }
                        )
    return settings


def setting_key(setting: dict) -> str:
    """Return a stable identifier for one OOF selection setting."""
    return (
        f"quality={setting['quality_cutoff']:.2f},"
        f"link={str(setting['linking_enabled']).lower()},"
        f"distance={setting['maximum_endpoint_distance']},"
        f"cosine={setting['minimum_tangent_cosine']:.2f},"
        f"gap={setting['minimum_gap_probability']:.2f}"
    )


def closest_endpoint_pair(first: dict, second: dict) -> tuple[float, tuple, tuple]:
    """Return the closest pair among two endpoints from each component."""
    candidates = []
    for first_endpoint in first["endpoints"]:
        for second_endpoint in second["endpoints"]:
            distance = math.dist(first_endpoint, second_endpoint)
            candidates.append((distance, first_endpoint, second_endpoint))
    return min(candidates, key=lambda item: item[0])


def segment_stays_inside_disk(
    first_endpoint: tuple[int, int],
    second_endpoint: tuple[int, int],
    disk: np.ndarray,
    *,
    line_thickness: int,
) -> bool:
    """Conservatively verify a sampled thick segment remains inside the disk."""
    if line_thickness < 1:
        raise ValueError("line_thickness must be positive.")
    distance = math.dist(first_endpoint, second_endpoint)
    sample_count = max(int(math.ceil(distance)) * 2 + 1, 2)
    x_values = np.rint(np.linspace(first_endpoint[0], second_endpoint[0], sample_count)).astype(int)
    y_values = np.rint(np.linspace(first_endpoint[1], second_endpoint[1], sample_count)).astype(int)
    radius = max(line_thickness // 2, 0)
    for x_value, y_value in zip(x_values, y_values, strict=True):
        for y_offset in range(-radius, radius + 1):
            for x_offset in range(-radius, radius + 1):
                x_index = x_value + x_offset
                y_index = y_value + y_offset
                if (
                    y_index < 0
                    or y_index >= disk.shape[0]
                    or x_index < 0
                    or x_index >= disk.shape[1]
                    or not disk[y_index, x_index]
                ):
                    return False
    return True


def select_and_link_instances(
    components: list[dict],
    scores: np.ndarray,
    probability: np.ndarray,
    disk: np.ndarray,
    *,
    setting: dict,
    line_thickness: int,
) -> list[dict]:
    """Filter components, then conservatively pair fragments within the solar disk."""
    accepted = [
        component
        for component, score in zip(components, scores, strict=True)
        if score >= setting["quality_cutoff"]
    ]
    if not setting["linking_enabled"] or len(accepted) < 2:
        return accepted

    eligible_pairs = []
    for first_index, first in enumerate(accepted):
        for second_index in range(first_index + 1, len(accepted)):
            second = accepted[second_index]
            tangent_cosine = abs(float(np.dot(first["tangent"], second["tangent"])))
            if tangent_cosine < setting["minimum_tangent_cosine"]:
                continue
            distance, first_endpoint, second_endpoint = closest_endpoint_pair(first, second)
            if distance > setting["maximum_endpoint_distance"]:
                continue
            if not segment_stays_inside_disk(
                first_endpoint,
                second_endpoint,
                disk,
                line_thickness=line_thickness,
            ):
                continue
            line = np.zeros_like(disk, dtype=np.uint8)
            cv2.line(
                line,
                tuple(map(int, first_endpoint)),
                tuple(map(int, second_endpoint)),
                color=1,
                thickness=line_thickness,
            )
            line_pixels = line != 0
            if np.any(line_pixels & ~disk):
                continue
            gap = line_pixels & ~(first["mask"] != 0) & ~(second["mask"] != 0)
            gap_probability = float(probability[gap].mean()) if np.any(gap) else 1.0
            if gap_probability < setting["minimum_gap_probability"]:
                continue
            eligible_pairs.append(
                (
                    distance,
                    -gap_probability,
                    first_index,
                    second_index,
                    line,
                )
            )

    used = set()
    merged = {}
    for _, _, first_index, second_index, line in sorted(
        eligible_pairs,
        key=lambda item: item[:4],
    ):
        if first_index in used or second_index in used:
            continue
        used.update((first_index, second_index))
        merged_mask = (
            accepted[first_index]["mask"] | accepted[second_index]["mask"] | line
        ).astype(np.uint8)
        merged[first_index] = {
            "mask": merged_mask,
            "pixel_indices": np.flatnonzero(merged_mask),
        }
        merged[second_index] = None

    output = []
    for index, component in enumerate(accepted):
        if index in merged:
            if merged[index] is not None:
                output.append(merged[index])
        else:
            output.append(component)
    return output


def empty_aggregate() -> dict[str, float | int]:
    """Create one instance-diagnostic accumulator."""
    return {
        "images": 0,
        "penalized_dice_sum": 0.0,
        "matched_dice_sum": 0.0,
        "matched_iou_sum": 0.0,
        "matched": 0,
        "missed": 0,
        "extra": 0,
        "one_to_many": 0,
        "many_to_one": 0,
    }


def update_aggregate(aggregate: dict, diagnostic: dict) -> None:
    """Accumulate one annotation-set diagnostic."""
    aggregate["images"] += 1
    aggregate["penalized_dice_sum"] += diagnostic["penalized_dice"]
    for name in (
        "matched_dice_sum",
        "matched_iou_sum",
        "matched",
        "missed",
        "extra",
        "one_to_many",
        "many_to_one",
    ):
        aggregate[name] += diagnostic[name]


def summarize_aggregate(aggregate: dict, *, key: str) -> dict:
    """Compute comparable means while preserving raw counts."""
    summary = dict(aggregate)
    summary["candidate"] = key
    summary["mean_penalized_instance_dice"] = aggregate["penalized_dice_sum"] / aggregate["images"]
    summary["mean_matched_instance_dice"] = aggregate["matched_dice_sum"] / max(
        aggregate["matched"],
        1,
    )
    summary["mean_matched_instance_iou"] = aggregate["matched_iou_sum"] / max(
        aggregate["matched"],
        1,
    )
    return summary


def target_label_sets(observation: dict, *, original_size: int, model_size: int) -> list:
    """Rasterize all annotators for one observation only when needed."""
    return [
        rasterize_instance_labels(
            annotation_set["polygons"],
            original_size=original_size,
            model_size=model_size,
        )
        for annotation_set in observation["annotation_sets"]
    ]


def evaluate_selection_grid(
    observations: list[dict],
    probabilities: dict[str, np.ndarray],
    component_scores: dict[str, np.ndarray],
    *,
    image_directory: Path,
    config: dict,
) -> tuple[list[dict], dict]:
    """Select filtering/linking parameters using cross-fitted OOF scores only."""
    settings = linking_candidates(config)
    aggregates = {setting_key(setting): empty_aggregate() for setting in settings}
    settings_by_key = {setting_key(setting): setting for setting in settings}
    model_size = config["model"]["input_size"]
    original_size = config["competition"]["image_height"]

    for observation_index, observation in enumerate(observations, start=1):
        key = observation["observation_key"]
        image = load_grayscale_image(
            image_directory / Path(observation["file_name"]).name,
            model_size=model_size,
        )
        probability = probabilities[key].astype(np.float32)
        components, disk = extract_components(
            probability,
            image,
            config=config["candidate_pool"],
        )
        scores = component_scores[key]
        if len(components) != len(scores):
            raise RuntimeError(f"Component regeneration changed for {key}.")
        targets = target_label_sets(
            observation,
            original_size=original_size,
            model_size=model_size,
        )
        for candidate_key, setting in settings_by_key.items():
            instances = select_and_link_instances(
                components,
                scores,
                probability,
                disk,
                setting=setting,
                line_thickness=config["fragment_linking"]["line_thickness"],
            )
            for target_labels in targets:
                diagnostic = sparse_instance_diagnostic(
                    instances,
                    target_labels,
                    minimum_iou=config["selection"]["matching_min_iou"],
                )
                update_aggregate(aggregates[candidate_key], diagnostic)
        if observation_index % 50 == 0 or observation_index == len(observations):
            print(f"Evaluated OOF selection grid for {observation_index}/{len(observations)}.")

    summaries = [summarize_aggregate(aggregate, key=key) for key, aggregate in aggregates.items()]
    summaries.sort(
        key=lambda item: (
            item["mean_penalized_instance_dice"],
            -item["extra"],
            -item["missed"],
            not settings_by_key[item["candidate"]]["linking_enabled"],
        ),
        reverse=True,
    )
    selected_summary = summaries[0]
    return summaries, settings_by_key[selected_summary["candidate"]]


def promotion_gate(selected: dict, *, config: dict) -> dict:
    """Evaluate the predeclared promotion gate against experiment 003."""
    baseline = config["baseline"]
    gate = config["promotion_gate"]
    maximum_misses = baseline["missed_instances"] * (
        1.0 + gate["maximum_missed_instance_increase_fraction"]
    )
    maximum_extras = baseline["extra_instances"] * (
        1.0 - gate["minimum_extra_instance_reduction_fraction"]
    )
    checks = {
        "penalized_dice": (
            selected["mean_penalized_instance_dice"] >= gate["minimum_mean_penalized_instance_dice"]
        ),
        "extra_instances": selected["extra"] <= maximum_extras,
        "missed_instances": selected["missed"] <= maximum_misses,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_allowed_extras": maximum_extras,
        "maximum_allowed_misses": maximum_misses,
    }


def calibration_summary(targets: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Report fixed score bins without using test components."""
    bins = np.linspace(0.0, 1.0, 11)
    records = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if upper == 1.0:
            selected = (scores >= lower) & (scores <= upper)
        else:
            selected = (scores >= lower) & (scores < upper)
        if np.any(selected):
            records.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "components": int(selected.sum()),
                    "mean_score": float(scores[selected].mean()),
                    "mean_target_iou": float(targets[selected].mean()),
                }
            )
    return records


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode one fixed-size non-empty instance with pycocotools."""
    if not np.any(mask):
        raise ValueError("Cannot encode an empty predicted instance.")
    encoded = mask_utils.encode(np.asfortranarray((mask != 0).astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str) or not counts:
        raise ValueError("pycocotools returned invalid compressed RLE counts.")
    return counts


def validate_and_write_submission(
    rows: list[tuple[str, str]],
    *,
    known_image_ids: set[str],
    output_path: Path,
) -> None:
    """Validate identifiers, duplicate masks, and canonical 2048-square RLE."""
    if not rows:
        raise ValueError("Experiment 004 predicted no test instances.")
    seen_ids = set()
    masks_by_image: dict[str, set[str]] = defaultdict(set)
    for filament_id, counts in rows:
        image_id, separator, instance_text = filament_id.rpartition("_")
        if (
            not separator
            or image_id not in known_image_ids
            or not instance_text.isdigit()
            or int(instance_text) < 1
        ):
            raise ValueError(f"Invalid filament_id: {filament_id}")
        if filament_id in seen_ids:
            raise ValueError(f"Duplicate filament_id: {filament_id}")
        seen_ids.add(filament_id)
        if counts in masks_by_image[image_id]:
            raise ValueError(f"Duplicate mask in image {image_id}.")
        masks_by_image[image_id].add(counts)
        decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("ascii")})
        if not np.any(decoded) or encode_binary_mask(decoded) != counts:
            raise ValueError(f"Non-canonical or empty RLE for {filament_id}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def infer_oof_probabilities(
    observations: list[dict],
    assignments: dict[str, int],
    *,
    image_directory: Path,
    config: dict,
    device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Run each experiment-003 model only on its grouped held-out fold."""
    import torch

    probabilities = {}
    fold_records = []
    for fold_index in config["validation"]["fold_indices"]:
        checkpoint = resolve_checkpoint(config, fold_index)
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        fold_observations = [
            observation
            for observation in observations
            if assignments[observation["observation_key"]] == fold_index
        ]
        print(
            f"Fold {fold_index}: checkpoint={checkpoint.name}, "
            f"OOF observations={len(fold_observations)}"
        )
        started = time.time()
        for index, observation in enumerate(fold_observations, start=1):
            image = load_grayscale_image(
                image_directory / Path(observation["file_name"]).name,
                model_size=config["model"]["input_size"],
            )
            probabilities[observation["observation_key"]] = predict_image(
                model,
                image,
                device=device,
                augmentations=config["inference"]["test_time_augmentations"],
            )
            if index % 30 == 0 or index == len(fold_observations):
                print(f"Fold {fold_index} OOF inference {index}/{len(fold_observations)}.")
        fold_records.append(
            {
                "fold": fold_index,
                "checkpoint": checkpoint.name,
                "oof_observations": len(fold_observations),
                "runtime_seconds": time.time() - started,
            }
        )
        del model
        torch.cuda.empty_cache()
    if set(probabilities) != set(assignments):
        raise RuntimeError("OOF inference did not cover every physical observation exactly once.")
    return probabilities, fold_records


def build_component_table(
    observations: list[dict],
    probabilities: dict[str, np.ndarray],
    assignments: dict[str, int],
    *,
    image_directory: Path,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Build OOF-only ranker features and annotator-aware quality targets."""
    all_features = []
    all_targets = []
    all_folds = []
    all_observation_keys = []
    row_slices = {}
    row_start = 0
    for index, observation in enumerate(observations, start=1):
        key = observation["observation_key"]
        image = load_grayscale_image(
            image_directory / Path(observation["file_name"]).name,
            model_size=config["model"]["input_size"],
        )
        probability = probabilities[key].astype(np.float32)
        components, _ = extract_components(
            probability,
            image,
            config=config["candidate_pool"],
        )
        targets = target_label_sets(
            observation,
            original_size=config["competition"]["image_height"],
            model_size=config["model"]["input_size"],
        )
        quality_targets = component_targets(components, targets)
        component_features = [component["features"] for component in components]
        all_features.extend(component_features)
        all_targets.extend(quality_targets.tolist())
        all_folds.extend([assignments[key]] * len(components))
        all_observation_keys.extend([key] * len(components))
        row_slices[key] = slice(row_start, row_start + len(components))
        row_start += len(components)
        if index % 50 == 0 or index == len(observations):
            print(f"Built OOF component rows for {index}/{len(observations)} observations.")

    features = np.asarray(all_features, dtype=np.float64)
    targets = np.asarray(all_targets, dtype=np.float64)
    folds = np.asarray(all_folds, dtype=np.int64)
    observation_keys = np.asarray(all_observation_keys, dtype=object)
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("Invalid OOF component table.")
    return features, targets, folds, observation_keys, row_slices


def infer_test_ensemble(
    test_directory: Path,
    *,
    config: dict,
    device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Average five experiment-003 checkpoints on test data after OOF selection."""
    import torch

    test_paths = sorted(test_directory.glob("*.jpeg"))
    if not test_paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_directory}.")
    test_images = {
        path.stem: load_grayscale_image(path, model_size=config["model"]["input_size"])
        for path in test_paths
    }
    probability_sums = {
        image_id: np.zeros_like(image, dtype=np.float32) for image_id, image in test_images.items()
    }
    for fold_index in config["validation"]["fold_indices"]:
        checkpoint = resolve_checkpoint(config, fold_index)
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        for index, (image_id, image) in enumerate(test_images.items(), start=1):
            probability_sums[image_id] += predict_image(
                model,
                image,
                device=device,
                augmentations=config["inference"]["test_time_augmentations"],
            ).astype(np.float32)
            if index % 30 == 0 or index == len(test_images):
                print(f"Fold {fold_index} test inference {index}/{len(test_images)} observations.")
        del model
        torch.cuda.empty_cache()
    fold_count = len(config["validation"]["fold_indices"])
    return (
        {key: value / fold_count for key, value in probability_sums.items()},
        test_images,
    )


def test_rows(
    probabilities: dict[str, np.ndarray],
    test_images: dict[str, np.ndarray],
    ranker,
    *,
    selected_setting: dict,
    config: dict,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Apply the frozen ranker and OOF-selected settings to final test inference."""
    rows = []
    per_image_counts = {}
    output_height = config["submission"]["mask_height"]
    output_width = config["submission"]["mask_width"]
    for index, (image_id, image) in enumerate(test_images.items(), start=1):
        probability = probabilities[image_id]
        components, disk = extract_components(
            probability,
            image,
            config=config["candidate_pool"],
        )
        if components:
            features = np.stack([component["features"] for component in components])
            scores = np.clip(ranker.predict(features), 0.0, 1.0)
        else:
            scores = np.empty(0, dtype=np.float64)
        instances = select_and_link_instances(
            components,
            scores,
            probability,
            disk,
            setting=selected_setting,
            line_thickness=config["fragment_linking"]["line_thickness"],
        )
        seen_counts = set()
        for instance_number, instance in enumerate(instances, start=1):
            resized = cv2.resize(
                instance["mask"],
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )
            counts = encode_binary_mask(resized)
            if counts in seen_counts:
                raise ValueError(f"Duplicate predicted instance in {image_id}.")
            seen_counts.add(counts)
            rows.append((f"{image_id}_{instance_number}", counts))
        per_image_counts[image_id] = len(instances)
        print(f"Final output {index}/{len(test_images)}: {image_id}, instances={len(instances)}")
    return rows, per_image_counts


def main() -> None:
    """Cross-fit component quality, select on OOF data, then infer test masks."""
    import torch

    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    if config["model"]["checkpoint_training_enabled"]:
        raise ValueError("Experiment 004 must not modify experiment-003 checkpoints.")
    if config["model"]["competition_trained_checkpoint_source"] != "experiment-003":
        raise ValueError("Experiment 004 requires experiment-003 competition-only checkpoints.")
    seed_everything(config["experiment"]["seed"])
    data_root = resolve_data_root(config)
    annotation_path = data_root / config["data"]["train_annotations"]
    train_image_directory = data_root / config["data"]["train_images"]
    test_image_directory = data_root / config["data"]["test_images"]
    observations, annotation_set_count, annotation_count = load_observations(annotation_path)
    assignments, fold_fingerprint = assign_grouped_folds(
        observations,
        n_splits=config["validation"]["n_splits"],
        seed=config["validation"]["seed"],
    )
    if fold_fingerprint != config["validation"]["expected_fold_fingerprint"]:
        raise RuntimeError(
            "Grouped-fold fingerprint changed: "
            f"expected={config['validation']['expected_fold_fingerprint']}, "
            f"actual={fold_fingerprint}."
        )
    fold_indices = config["validation"]["fold_indices"]
    if sorted(fold_indices) != list(range(config["validation"]["n_splits"])):
        raise ValueError("Experiment 004 must cover every grouped fold exactly once.")

    config_sha256 = hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Official competition data root: {data_root}")
    print(f"Config SHA-256: {config_sha256}")
    print(f"Fold fingerprint: {fold_fingerprint}")
    print(f"Inference device: {device}")

    oof_probabilities, fold_records = infer_oof_probabilities(
        observations,
        assignments,
        image_directory=train_image_directory,
        config=config,
        device=device,
    )
    features, targets, folds, observation_keys, row_slices = build_component_table(
        observations,
        oof_probabilities,
        assignments,
        image_directory=train_image_directory,
        config=config,
    )
    scores, cross_fit_records = cross_fitted_scores(
        features,
        targets,
        folds,
        observation_keys,
        fold_indices=fold_indices,
        config=config["component_quality"],
    )
    scores_by_observation = {key: scores[row_slice] for key, row_slice in row_slices.items()}
    candidate_summaries, selected_setting = evaluate_selection_grid(
        observations,
        oof_probabilities,
        scores_by_observation,
        image_directory=train_image_directory,
        config=config,
    )
    selected_summary = candidate_summaries[0]
    gate_result = promotion_gate(selected_summary, config=config["selection"])
    print(
        "Selected OOF setting: "
        f"{selected_summary['candidate']}, "
        f"penalized_dice={selected_summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={selected_summary['missed']}, extra={selected_summary['extra']}, "
        f"promotion_gate_passed={gate_result['passed']}"
    )

    final_ranker = make_ranker(config["component_quality"])
    final_ranker.fit(features, targets)
    del oof_probabilities

    test_probabilities, test_images = infer_test_ensemble(
        test_image_directory,
        config=config,
        device=device,
    )
    rows, per_image_counts = test_rows(
        test_probabilities,
        test_images,
        final_ranker,
        selected_setting=selected_setting,
        config=config,
    )
    submission_path = Path(config["submission"]["output_path"])
    validate_and_write_submission(
        rows,
        known_image_ids=set(test_images),
        output_path=submission_path,
    )

    metadata = {
        "experiment_id": config["experiment"]["id"],
        "config_sha256": config_sha256,
        "fold_fingerprint": fold_fingerprint,
        "physical_observations": len(observations),
        "annotation_sets": annotation_set_count,
        "annotations": annotation_count,
        "feature_names": FEATURE_NAMES,
        "component_rows": len(targets),
        "component_target_mean": float(targets.mean()),
        "component_cross_fitted_score_mean": float(scores.mean()),
        "component_cross_fitted_mean_absolute_error": float(np.mean(np.abs(scores - targets))),
        "component_quality_calibration": calibration_summary(targets, scores),
        "cross_fit_folds": cross_fit_records,
        "base_model_oof_inference": fold_records,
        "selection_candidates": candidate_summaries,
        "selected_setting": selected_setting,
        "selected_oof_diagnostics": selected_summary,
        "promotion_gate": gate_result,
        "checkpoint_source": config["model"]["competition_trained_checkpoint_source"],
        "checkpoint_training_enabled": config["model"]["checkpoint_training_enabled"],
        "external_labeled_data": config["model"]["external_labeled_data"],
        "pretrained_weights": config["model"]["pretrained_weights"],
        "test_statistics_used_for_selection": False,
        "test_time_augmentations": config["inference"]["test_time_augmentations"],
        "ensemble_folds": fold_indices,
        "test_images": len(test_images),
        "submission_rows": len(rows),
        "predicted_instances_per_image": per_image_counts,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-004-run-metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Submission candidate written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This private kernel creates artifacts only; it does not submit to the competition.")


if __name__ == "__main__":
    main()
