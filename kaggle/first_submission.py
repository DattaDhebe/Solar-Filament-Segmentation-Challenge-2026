"""Kaggle GPU script for experiment-003, OOF TTA instance refinement.

This file is deliberately self-contained because the Kaggle kernels API uploads
only the configured code file. Keep EMBEDDED_CONFIG_YAML synchronized with
configs/experiment-003-oof-tta-refinement.yaml; a synthetic test enforces that.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
  id: experiment-003
  name: oof-tta-instance-refinement
  seed: 20260730
  output_root: outputs/experiments
  mode: checkpoint-finetuning-and-refinement

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
  checkpoint_root: /kaggle/input/solar-filament-overnight-ensemble

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

model:
  task: instance-segmentation
  architecture: small-unet
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 24
  pretrained_weights: false
  external_labeled_data: false
  competition_trained_checkpoint_source: experiment-002

training:
  target_strategy: annotator-soft-consensus
  epochs: 18
  minimum_epochs: 8
  early_stopping_patience: 4
  batch_size: 2
  learning_rate: 0.0002
  weight_decay: 0.0001
  bce_positive_weight: 4.0
  dice_loss_weight: 1.0
  bce_loss_weight: 1.0
  num_workers: 0
  mixed_precision: true
  augmentations:
    horizontal_flip_probability: 0.5
    vertical_flip_probability: 0.5
    rotate_90_probability: 0.5
    brightness_delta: 0.05
    contrast_range:
      - 0.9
      - 1.1

inference:
  batch_size: 2
  test_time_augmentations:
    - identity
    - horizontal-flip
    - vertical-flip
    - horizontal-vertical-flip

postprocessing:
  probability_threshold: 0.6
  closing_kernel: 5
  closing_iterations: 1
  disk_erosion_pixels: 8
  min_component_area_at_model_resolution: 32
  max_component_area_at_model_resolution: 40000
  minimum_component_mean_probability: 0.0
  connectivity: 8
  matching_min_iou: 0.1
  selection_grid:
    probability_thresholds:
      - 0.5
      - 0.6
      - 0.7
    closing_kernels:
      - 5
      - 7
    minimum_component_areas:
      - 32
      - 64
      - 96
      - 128
    minimum_component_mean_probabilities:
      - 0.0
      - 0.7
      - 0.8

metric:
  primary: mean-matched-instance-dice
  diagnostics:
    - semantic-dice-distribution
    - matched-instance-dice
    - matched-instance-iou
    - missed-and-extra-instances
    - one-to-many-relations
    - many-to-one-relations

submission:
  columns:
    - filament_id
    - segmentation_rle
  rle_format: coco-compressed-counts
  mask_height: 2048
  mask_width: 2048
  output_path: /kaggle/working/experiment-003-submission.csv
""".strip()


def seed_everything(seed: int) -> None:
    """Make data ordering, augmentation, and model initialization reproducible."""
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

    input_root = Path("/kaggle/input")
    candidates = sorted(input_root.glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
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
    """Resolve only experiment-002 checkpoints produced from competition data."""
    file_name = f"experiment-002-fold-{fold_index}.pt"
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


def load_examples(annotation_path: Path) -> tuple[list[dict], int]:
    """Load train-only COCO records and associate polygons with image records."""
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    annotations_by_image: dict[object, list[list[float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("This pipeline supports the supplied polygon annotations only.")
        annotations_by_image[annotation["image_id"]].extend(segmentation)

    examples = []
    for image_record in document["images"]:
        file_name = str(image_record["file_name"])
        polygons = annotations_by_image.get(image_record["id"], [])
        examples.append(
            {
                "image_record_id": image_record["id"],
                "file_name": file_name,
                "observation_key": Path(file_name).stem,
                "polygons": polygons,
            }
        )
    return examples, len(document["annotations"])


def assign_grouped_folds(
    examples: list[dict],
    *,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, int], str]:
    """Assign every physical observation to one deterministic validation fold."""
    observation_keys = sorted({str(example["observation_key"]) for example in examples})
    if n_splits < 2 or n_splits > len(observation_keys):
        raise ValueError("n_splits must be between 2 and the physical-observation count.")

    random.Random(seed).shuffle(observation_keys)
    assignments = {key: index % n_splits for index, key in enumerate(observation_keys)}
    canonical = "\n".join(f"{key},{assignments[key]}" for key in sorted(assignments))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return assignments, fingerprint


def rasterize_instances(
    polygons: list[list[float]],
    *,
    original_size: int,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize official COCO polygons as semantic and instance-label masks."""
    if not polygons:
        shape = (model_size, model_size)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint16)

    scale = model_size / original_size
    scaled_polygons = [[coordinate * scale for coordinate in polygon] for polygon in polygons]
    rles = mask_utils.frPyObjects(scaled_polygons, model_size, model_size)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, np.newaxis]
    semantic = np.any(decoded != 0, axis=2).astype(np.uint8)
    instance_labels = np.zeros((model_size, model_size), dtype=np.uint16)
    for index in range(decoded.shape[2]):
        instance_labels[decoded[:, :, index] != 0] = index + 1
    return semantic, instance_labels


def prepare_examples(
    examples: list[dict],
    *,
    image_directory: Path,
    original_size: int,
    model_size: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Cache grayscale images and annotation-set masks at model resolution."""
    image_cache: dict[str, np.ndarray] = {}
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    resampling = Image.Resampling

    for index, example in enumerate(examples, start=1):
        file_name = str(example["file_name"])
        if file_name not in image_cache:
            image_path = image_directory / Path(file_name).name
            with Image.open(image_path) as image:
                if image.mode != "L":
                    image = image.convert("L")
                resized = image.resize((model_size, model_size), resampling.BILINEAR)
                image_cache[file_name] = np.asarray(resized, dtype=np.uint8)

        semantic_mask, instance_labels = rasterize_instances(
            example["polygons"],
            original_size=original_size,
            model_size=model_size,
        )
        prepared.append((image_cache[file_name], semantic_mask != 0, instance_labels))

        if index % 100 == 0 or index == len(examples):
            print(f"Prepared {index}/{len(examples)} annotation sets.")
    return prepared


def consensus_training_samples(
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    examples: list[dict],
    assignments: dict[str, int],
    *,
    validation_fold: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Average annotator foreground targets once per physical training observation."""
    indices_by_observation: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        observation_key = str(example["observation_key"])
        if assignments[observation_key] != validation_fold:
            indices_by_observation[observation_key].append(index)

    consensus_samples = []
    unused_instance_labels = np.empty((0, 0), dtype=np.uint16)
    for observation_key in sorted(indices_by_observation):
        indices = indices_by_observation[observation_key]
        image = prepared[indices[0]][0]
        target = np.zeros_like(prepared[indices[0]][1], dtype=np.float32)
        for index in indices:
            if not np.array_equal(image, prepared[index][0]):
                raise RuntimeError(f"Annotator records disagree on pixels for {observation_key}.")
            target += prepared[index][1]
        target /= len(indices)
        consensus_samples.append((image, target, unused_instance_labels))
    return consensus_samples


def build_model(base_channels: int):
    """Construct a compact U-Net with random initialization."""
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


def make_dataset_class():
    """Create the torch Dataset class without requiring torch during source checks."""
    import torch

    class FilamentDataset(torch.utils.data.Dataset):
        def __init__(self, samples, *, augment: bool, augmentation_config: dict) -> None:
            self.samples = samples
            self.augment = augment
            self.augmentation_config = augmentation_config

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            image, mask, _ = self.samples[index]
            image = image.copy()
            mask = mask.copy()
            if self.augment:
                if random.random() < self.augmentation_config["horizontal_flip_probability"]:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)
                if random.random() < self.augmentation_config["vertical_flip_probability"]:
                    image = np.flipud(image)
                    mask = np.flipud(mask)
                if random.random() < self.augmentation_config["rotate_90_probability"]:
                    turns = random.randint(1, 3)
                    image = np.rot90(image, turns)
                    mask = np.rot90(mask, turns)
                contrast_min, contrast_max = self.augmentation_config["contrast_range"]
                contrast = random.uniform(contrast_min, contrast_max)
                brightness = random.uniform(
                    -self.augmentation_config["brightness_delta"],
                    self.augmentation_config["brightness_delta"],
                )
                image = np.clip(
                    image.astype(np.float32) * contrast + brightness * 255.0,
                    0,
                    255,
                ).astype(np.uint8)

            image_tensor = (
                torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0
            )
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
            return image_tensor, mask_tensor

    return FilamentDataset


def combined_loss(logits, targets, *, config: dict):
    """Use fixed weighted BCE plus soft Dice loss."""
    import torch
    from torch.nn import functional as functional

    positive_weight = torch.tensor(
        config["bce_positive_weight"],
        device=logits.device,
    )
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )
    probabilities = torch.sigmoid(logits)
    reduce_dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduce_dimensions)
    denominator = probabilities.sum(dim=reduce_dimensions) + targets.sum(dim=reduce_dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return config["bce_loss_weight"] * bce + config["dice_loss_weight"] * dice_loss


def semantic_dice(probabilities, targets, *, threshold: float) -> np.ndarray:
    """Return per-image semantic Dice for a validation diagnostic."""
    predictions = probabilities >= threshold
    expected = targets >= 0.5
    intersection = (predictions & expected).sum(axis=(1, 2, 3))
    denominator = predictions.sum(axis=(1, 2, 3)) + expected.sum(axis=(1, 2, 3))
    return np.where(denominator == 0, 1.0, (2.0 * intersection) / denominator)


def train_model(model, train_loader, validation_loader, *, config: dict, device):
    """Train one fixed grouped fold and retain its best semantic-Dice checkpoint."""
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )
    amp_enabled = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    threshold = config["postprocessing"]["probability_threshold"]
    best_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(logits, masks, config=config["training"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_scores = []
        with torch.no_grad():
            for images, masks in validation_loader:
                images = images.to(device, non_blocking=True)
                probabilities = torch.sigmoid(model(images)).cpu().numpy()
                validation_scores.extend(
                    semantic_dice(
                        probabilities,
                        masks.numpy(),
                        threshold=threshold,
                    ).tolist()
                )

        mean_loss = float(np.mean(losses))
        mean_score = float(np.mean(validation_scores))
        history.append(
            {
                "epoch": epoch,
                "training_loss": mean_loss,
                "validation_semantic_dice": mean_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Epoch {epoch:02d}/{config['training']['epochs']}: "
            f"loss={mean_loss:.5f}, validation_semantic_dice={mean_score:.5f}"
        )
        if mean_score > best_score:
            best_score = mean_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if (
            epoch >= config["training"]["minimum_epochs"]
            and epochs_without_improvement >= config["training"]["early_stopping_patience"]
        ):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_score, best_epoch, history


def solar_disk_mask(image: np.ndarray, *, erosion_pixels: int) -> np.ndarray:
    """Estimate the disk from the image itself and erode the bright limb."""
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    label_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        thresholded,
        connectivity=8,
    )
    if label_count <= 1:
        return np.ones_like(image, dtype=bool)
    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    disk = (labels == largest_label).astype(np.uint8)
    if erosion_pixels > 0:
        kernel_size = erosion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        disk = cv2.erode(disk, kernel, iterations=1)
    return disk != 0


def separate_instances(
    probability: np.ndarray,
    grayscale_image: np.ndarray,
    *,
    config: dict,
) -> list[np.ndarray]:
    """Threshold semantic foreground and preserve connected components as instances."""
    foreground = probability >= config["probability_threshold"]
    foreground &= solar_disk_mask(
        grayscale_image,
        erosion_pixels=config["disk_erosion_pixels"],
    )
    kernel_size = config["closing_kernel"]
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config["closing_iterations"],
        )

    label_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=config["connectivity"],
    )
    candidates = []
    for label in range(1, label_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if not (
            config["min_component_area_at_model_resolution"]
            <= area
            <= config["max_component_area_at_model_resolution"]
        ):
            continue
        component = labels == label
        mean_probability = float(probability[component].mean())
        if mean_probability < config["minimum_component_mean_probability"]:
            continue
        centroid_x, centroid_y = centroids[label]
        candidates.append((-area, float(centroid_y), float(centroid_x), label))

    candidates.sort()
    return [(labels == label).astype(np.uint8) for _, _, _, label in candidates]


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode one fixed-size binary instance with pycocotools."""
    encoded = mask_utils.encode(np.asfortranarray((mask != 0).astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str) or not counts:
        raise ValueError("pycocotools returned invalid compressed RLE counts.")
    return counts


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


def predict_validation_probabilities(
    model,
    validation_loader,
    *,
    device,
    augmentations: list[str],
) -> list[np.ndarray]:
    """Predict one held-out fold in its stable dataset order."""
    import torch

    predictions = []
    model.eval()
    with torch.no_grad():
        for images, _ in validation_loader:
            probabilities = predict_with_tta(
                model,
                images.to(device, non_blocking=True),
                augmentations=augmentations,
            )
            predictions.extend(probabilities[:, 0].cpu().numpy())
    return predictions


def instance_diagnostic(
    predicted_instances: list[np.ndarray],
    target_labels: np.ndarray,
    *,
    minimum_iou: float,
) -> dict[str, float | int]:
    """Greedily match instances and penalize both missed and extra predictions."""
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

    predicted_count = len(predicted_instances)
    target_count = len(target_ids)
    denominator = max(predicted_count, target_count, 1)
    return {
        "penalized_dice": float(sum(matched_dice) / denominator),
        "matched_dice_sum": float(sum(matched_dice)),
        "matched_iou_sum": float(sum(matched_iou)),
        "matched": len(matched_dice),
        "missed": target_count - len(matched_targets),
        "extra": predicted_count - len(matched_predictions),
        "one_to_many": sum(count > 1 for count in overlap_by_target.values()),
        "many_to_one": sum(count > 1 for count in overlap_by_prediction),
    }


def postprocessing_candidates(config: dict) -> list[dict]:
    """Expand the versioned validation-only post-processing grid."""
    grid = config["selection_grid"]
    candidates = []
    for threshold in grid["probability_thresholds"]:
        for closing_kernel in grid["closing_kernels"]:
            for minimum_area in grid["minimum_component_areas"]:
                for minimum_mean_probability in grid["minimum_component_mean_probabilities"]:
                    candidate = dict(config)
                    candidate["probability_threshold"] = threshold
                    candidate["closing_kernel"] = closing_kernel
                    candidate["min_component_area_at_model_resolution"] = minimum_area
                    candidate["minimum_component_mean_probability"] = minimum_mean_probability
                    candidates.append(candidate)
    return candidates


def candidate_key(config: dict) -> str:
    """Return a stable compact identifier for one post-processing candidate."""
    return (
        f"threshold={config['probability_threshold']:.2f},"
        f"closing={config['closing_kernel']},"
        f"min_area={config['min_component_area_at_model_resolution']},"
        f"min_mean={config['minimum_component_mean_probability']:.2f}"
    )


def evaluate_postprocessing_grid(
    probabilities: list[np.ndarray],
    validation_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    config: dict,
) -> dict[str, dict[str, float | int]]:
    """Evaluate candidate instance separation on grouped OOF predictions only."""
    totals = {}
    for candidate in postprocessing_candidates(config):
        key = candidate_key(candidate)
        aggregate: dict[str, float | int] = {
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
        for probability, (image, _, target_labels) in zip(
            probabilities,
            validation_samples,
            strict=True,
        ):
            predicted_instances = separate_instances(probability, image, config=candidate)
            diagnostic = instance_diagnostic(
                predicted_instances,
                target_labels,
                minimum_iou=config["matching_min_iou"],
            )
            aggregate["images"] += 1
            for name, value in diagnostic.items():
                aggregate[f"{name}_sum" if name == "penalized_dice" else name] += value
        totals[key] = aggregate
        mean_score = aggregate["penalized_dice_sum"] / aggregate["images"]
        print(f"OOF candidate {key}: penalized_instance_dice={mean_score:.5f}")
    return totals


def merge_candidate_totals(
    destination: dict[str, dict[str, float | int]],
    source: dict[str, dict[str, float | int]],
) -> None:
    """Accumulate candidate diagnostics across held-out folds."""
    for key, values in source.items():
        aggregate = destination.setdefault(key, {name: 0 for name in values})
        for name, value in values.items():
            aggregate[name] += value


def prepare_test_images(test_directory: Path, *, model_size: int) -> dict[str, np.ndarray]:
    """Load final-inference images once as grayscale model inputs."""
    test_paths = sorted(test_directory.glob("*.jpeg"))
    if not test_paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_directory}.")
    prepared = {}
    for image_path in test_paths:
        with Image.open(image_path) as image:
            resized = image.convert("L").resize(
                (model_size, model_size),
                Image.Resampling.BILINEAR,
            )
            prepared[image_path.stem] = np.asarray(resized, dtype=np.uint8)
    return prepared


def accumulate_test_probabilities(
    model,
    test_images: dict[str, np.ndarray],
    probability_sums: dict[str, np.ndarray],
    *,
    device,
    augmentations: list[str],
) -> None:
    """Add one fold model's final-inference probabilities to the ensemble."""
    import torch

    model.eval()
    with torch.no_grad():
        for index, (image_id, image_array) in enumerate(test_images.items(), start=1):
            tensor = (
                torch.from_numpy(image_array.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                / 255.0
            )
            probability = (
                predict_with_tta(
                    model,
                    tensor,
                    augmentations=augmentations,
                )[0, 0]
                .cpu()
                .numpy()
            )
            if image_id not in probability_sums:
                probability_sums[image_id] = probability.astype(np.float32)
            else:
                probability_sums[image_id] += probability
            if index % 30 == 0 or index == len(test_images):
                print(f"Accumulated ensemble inference for {index}/{len(test_images)} images.")


def rows_from_ensemble(
    probability_sums: dict[str, np.ndarray],
    test_images: dict[str, np.ndarray],
    *,
    fold_count: int,
    postprocessing_config: dict,
    submission_config: dict,
):
    """Average fold probabilities and encode separated full-resolution instances."""
    output_height = submission_config["mask_height"]
    output_width = submission_config["mask_width"]
    rows = []
    per_image_counts = {}
    for index, (image_id, image_array) in enumerate(test_images.items(), start=1):
        probability = probability_sums[image_id] / fold_count
        instances = separate_instances(
            probability,
            image_array,
            config=postprocessing_config,
        )

        seen_counts = set()
        image_rows = []
        for instance in instances:
            resized_instance = cv2.resize(
                instance,
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )
            counts = encode_binary_mask(resized_instance)
            if counts in seen_counts:
                raise ValueError(f"Duplicate predicted instance in {image_id}.")
            seen_counts.add(counts)
            image_rows.append((f"{image_id}_{len(image_rows) + 1}", counts))
        rows.extend(image_rows)
        per_image_counts[image_id] = len(image_rows)
        print(
            f"Ensemble output {index}/{len(test_images)}: {image_id}, instances={len(image_rows)}"
        )

    if not rows:
        raise ValueError("The ensemble predicted no instances in the complete test set.")
    return rows, per_image_counts


def validate_and_write_submission(
    rows: list[tuple[str, str]],
    *,
    known_image_ids: set[str],
    output_path: Path,
) -> None:
    """Perform identifier/RLE checks and write exactly the required two columns."""
    seen_ids = set()
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
        if (
            not counts
            or not counts.isascii()
            or any(character in counts for character in ('"', "'", "\r", "\n"))
        ):
            raise ValueError(f"Invalid COCO compressed RLE for {filament_id}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def main() -> None:
    """Fine-tune grouped folds, refine OOF post-processing, and ensemble test inference."""
    import torch

    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    data_root = resolve_data_root(config)
    print(f"Official competition data root: {data_root}")
    print(f"Config SHA-256: {hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest()}")

    annotation_path = data_root / config["data"]["train_annotations"]
    train_image_directory = data_root / config["data"]["train_images"]
    test_image_directory = data_root / config["data"]["test_images"]
    examples, annotation_count = load_examples(annotation_path)
    assignments, fold_fingerprint = assign_grouped_folds(
        examples,
        n_splits=config["validation"]["n_splits"],
        seed=config["validation"]["seed"],
    )
    print(f"Fold fingerprint={fold_fingerprint}")
    prepared = prepare_examples(
        examples,
        image_directory=train_image_directory,
        original_size=config["competition"]["image_height"],
        model_size=config["model"]["input_size"],
    )
    dataset_class = make_dataset_class()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training and inference device: {device}")
    fold_indices = config["validation"]["fold_indices"]
    if sorted(fold_indices) != list(range(config["validation"]["n_splits"])):
        raise ValueError("Experiment-003 must cover every configured grouped fold exactly once.")

    test_images = prepare_test_images(
        test_image_directory,
        model_size=config["model"]["input_size"],
    )
    probability_sums: dict[str, np.ndarray] = {}
    all_candidate_totals: dict[str, dict[str, float | int]] = {}
    fold_results = []

    for fold_index in fold_indices:
        fold_seed = seed + fold_index
        seed_everything(fold_seed)
        if config["training"]["target_strategy"] != "annotator-soft-consensus":
            raise ValueError("Experiment-003 requires annotator-soft-consensus targets.")
        training_samples = consensus_training_samples(
            prepared,
            examples,
            assignments,
            validation_fold=fold_index,
        )
        validation_samples = [
            sample
            for sample, example in zip(prepared, examples, strict=True)
            if assignments[str(example["observation_key"])] == fold_index
        ]
        print(
            f"Fold {fold_index}: train annotation sets={len(training_samples)}, "
            f"validation annotation sets={len(validation_samples)}"
        )

        training_dataset = dataset_class(
            training_samples,
            augment=True,
            augmentation_config=config["training"]["augmentations"],
        )
        validation_dataset = dataset_class(
            validation_samples,
            augment=False,
            augmentation_config=config["training"]["augmentations"],
        )
        train_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(fold_seed),
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config["inference"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model(config["model"]["base_channels"]).to(device)
        checkpoint_path = resolve_checkpoint(config, fold_index)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded competition-trained checkpoint: {checkpoint_path}")
        model, best_score, best_epoch, history = train_model(
            model,
            train_loader,
            validation_loader,
            config=config,
            device=device,
        )
        fine_tuned_checkpoint = Path(f"/kaggle/working/experiment-003-fold-{fold_index}.pt")
        torch.save(model.state_dict(), fine_tuned_checkpoint)

        validation_probabilities = predict_validation_probabilities(
            model,
            validation_loader,
            device=device,
            augmentations=config["inference"]["test_time_augmentations"],
        )
        validation_semantic_scores = [
            float(
                semantic_dice(
                    probability[np.newaxis, np.newaxis],
                    semantic_mask[np.newaxis, np.newaxis],
                    threshold=0.5,
                )[0]
            )
            for probability, (_, semantic_mask, _) in zip(
                validation_probabilities,
                validation_samples,
                strict=True,
            )
        ]
        fold_candidate_totals = evaluate_postprocessing_grid(
            validation_probabilities,
            validation_samples,
            config=config["postprocessing"],
        )
        merge_candidate_totals(all_candidate_totals, fold_candidate_totals)
        accumulate_test_probabilities(
            model,
            test_images,
            probability_sums,
            device=device,
            augmentations=config["inference"]["test_time_augmentations"],
        )
        fold_results.append(
            {
                "fold": fold_index,
                "seed": fold_seed,
                "training_annotation_sets": len(training_samples),
                "validation_annotation_sets": len(validation_samples),
                "fine_tuning_best_epoch": best_epoch,
                "best_fine_tuning_semantic_dice": best_score,
                "tta_validation_semantic_dice": float(np.mean(validation_semantic_scores)),
                "source_checkpoint": checkpoint_path.name,
                "fine_tuned_checkpoint": fine_tuned_checkpoint.name,
                "training_history": history,
            }
        )
        del model, train_loader, validation_loader, validation_probabilities
        torch.cuda.empty_cache()

    candidate_summaries = []
    for key, totals in all_candidate_totals.items():
        summary = dict(totals)
        summary["candidate"] = key
        summary["mean_penalized_instance_dice"] = totals["penalized_dice_sum"] / totals["images"]
        summary["mean_matched_instance_dice"] = totals["matched_dice_sum"] / max(
            totals["matched"], 1
        )
        summary["mean_matched_instance_iou"] = totals["matched_iou_sum"] / max(totals["matched"], 1)
        candidate_summaries.append(summary)
    candidate_summaries.sort(
        key=lambda item: item["mean_penalized_instance_dice"],
        reverse=True,
    )
    selected_summary = candidate_summaries[0]
    selected_config = next(
        candidate
        for candidate in postprocessing_candidates(config["postprocessing"])
        if candidate_key(candidate) == selected_summary["candidate"]
    )
    print(
        "Selected OOF post-processing: "
        f"{selected_summary['candidate']}, "
        f"penalized_instance_dice={selected_summary['mean_penalized_instance_dice']:.5f}"
    )

    rows, per_image_counts = rows_from_ensemble(
        probability_sums,
        test_images,
        fold_count=len(fold_indices),
        postprocessing_config=selected_config,
        submission_config=config["submission"],
    )
    known_image_ids = set(test_images)
    submission_path = Path(config["submission"]["output_path"])
    validate_and_write_submission(
        rows,
        known_image_ids=known_image_ids,
        output_path=submission_path,
    )

    metadata = {
        "experiment_id": config["experiment"]["id"],
        "config_sha256": hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest(),
        "fold_fingerprint": fold_fingerprint,
        "physical_observations": len(assignments),
        "annotation_sets": len(examples),
        "annotations": annotation_count,
        "folds": fold_results,
        "mean_best_fine_tuning_semantic_dice": float(
            np.mean([result["best_fine_tuning_semantic_dice"] for result in fold_results])
        ),
        "mean_tta_fold_semantic_dice": float(
            np.mean([result["tta_validation_semantic_dice"] for result in fold_results])
        ),
        "postprocessing_candidates": candidate_summaries,
        "selected_postprocessing": {
            "candidate": selected_summary["candidate"],
            "mean_penalized_instance_dice": selected_summary["mean_penalized_instance_dice"],
            "mean_matched_instance_dice": selected_summary["mean_matched_instance_dice"],
            "mean_matched_instance_iou": selected_summary["mean_matched_instance_iou"],
            "matched": selected_summary["matched"],
            "missed": selected_summary["missed"],
            "extra": selected_summary["extra"],
            "one_to_many": selected_summary["one_to_many"],
            "many_to_one": selected_summary["many_to_one"],
        },
        "semantic_training_only": True,
        "instance_separation": "connected-components",
        "checkpoint_source": config["model"]["competition_trained_checkpoint_source"],
        "test_time_augmentations": config["inference"]["test_time_augmentations"],
        "ensemble_folds": fold_indices,
        "test_images": len(known_image_ids),
        "submission_rows": len(rows),
        "predicted_instances_per_image": per_image_counts,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-003-run-metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Submission written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This kernel creates an artifact only; it does not submit to the competition.")


if __name__ == "__main__":
    main()
