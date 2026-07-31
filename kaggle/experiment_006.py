"""Private Kaggle GPU script for experiment-006 OOF blend calibration.

The script reconstructs grouped out-of-fold probabilities from the immutable
experiment-003 and experiment-005 checkpoints, selects one probability blend
and post-processing setting using only OOF instance diagnostics, verifies the
two reference candidates, and only then performs final test inference. Test
observations never influence calibration, model selection, or promotion.
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
  id: experiment-006
  name: oof-blend-calibration
  seed: 20260801
  output_root: outputs/experiments
  mode: checkpoint-blend-and-oof-calibration

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
  checkpoint_roots:
    experiment-003: /kaggle/input/solar-filament-oof-tta-refinement
    experiment-005: /kaggle/input/solar-filament-hybrid-annotator

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

model:
  task: instance-segmentation
  architecture: small-unet
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 24
  pretrained_weights: false
  external_labeled_data: false
  competition_trained_checkpoint_sources:
    - experiment-003
    - experiment-005

inference:
  batch_size: 1
  test_time_augmentations:
    - identity
    - horizontal-flip
    - vertical-flip
    - horizontal-vertical-flip

calibration:
  selection_metric: mean-penalized-instance-dice
  experiment_005_probability_weights:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
  probability_thresholds:
    - 0.5
    - 0.525
    - 0.55
  closing_kernels:
    - 7
  minimum_component_areas:
    - 96
    - 128
    - 160
  minimum_component_mean_probabilities:
    - 0.8
    - 0.825
    - 0.85
  feasibility:
    maximum_missed_instances: 1692
    maximum_extra_instances: 2103
  reference_candidates:
    experiment-003:
      experiment_005_probability_weight: 0.0
      probability_threshold: 0.5
      closing_kernel: 7
      min_component_area_at_model_resolution: 96
      minimum_component_mean_probability: 0.8
      expected_mean_penalized_instance_dice: 0.490129
      expected_missed_instances: 1693
      expected_extra_instances: 2103
    experiment-005:
      experiment_005_probability_weight: 1.0
      probability_threshold: 0.5
      closing_kernel: 7
      min_component_area_at_model_resolution: 96
      minimum_component_mean_probability: 0.8
      expected_mean_penalized_instance_dice: 0.490488
      expected_missed_instances: 1609
      expected_extra_instances: 2309
    dice_absolute_tolerance: 0.00001

postprocessing:
  probability_threshold: 0.5
  closing_kernel: 7
  closing_iterations: 1
  disk_erosion_pixels: 8
  min_component_area_at_model_resolution: 96
  max_component_area_at_model_resolution: 40000
  minimum_component_mean_probability: 0.8
  connectivity: 8
  matching_min_iou: 0.1

promotion:
  baseline_experiment: experiment-003
  baseline_mean_penalized_instance_dice: 0.490129
  baseline_missed_instances: 1693
  baseline_extra_instances: 2103
  minimum_mean_penalized_instance_dice: 0.495
  maximum_missed_instances: 1692
  maximum_extra_instances: 2103

metric:
  primary: mean-matched-instance-dice
  diagnostics:
    - semantic-dice-distribution
    - matched-instance-dice
    - matched-instance-iou
    - missed-and-extra-instances
    - one-to-many-relations
    - many-to-one-relations
    - single-versus-multi-annotator-dice
    - missed-instances-by-annotator-count

submission:
  columns:
    - filament_id
    - segmentation_rle
  rle_format: coco-compressed-counts
  mask_height: 2048
  mask_width: 2048
  output_path: /kaggle/working/experiment-006-submission.csv
""".strip()


def seed_everything(seed: int) -> None:
    """Make model training and augmentation reproducible."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def deterministic_annotator_index(
    seed: int,
    epoch: int,
    sample_index: int,
    annotator_count: int,
) -> int:
    """Select one annotator deterministically while rotating across epochs."""
    if annotator_count < 1:
        raise ValueError("Every training observation needs at least one annotator.")
    mixed = seed + 1_000_003 * epoch + 97_409 * sample_index
    return random.Random(mixed).randrange(annotator_count)


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
            "Could not uniquely resolve official competition data; "
            f"configured={configured}, candidates={valid}."
        )
    return valid[0]


def resolve_checkpoint(config: dict, source: str, fold_index: int) -> Path:
    """Resolve one competition-trained checkpoint from an approved source."""
    approved = config["model"]["competition_trained_checkpoint_sources"]
    if source not in approved:
        raise ValueError(f"Checkpoint source is not approved: {source}")
    file_name = f"{source}-fold-{fold_index}.pt"
    configured = Path(config["data"]["checkpoint_roots"][source]) / file_name
    if configured.is_file():
        return configured
    candidates = sorted(Path("/kaggle/input").glob(f"**/{file_name}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {file_name}; "
            f"configured={configured}, candidates={candidates}."
        )
    return candidates[0]


def load_observation_records(annotation_path: Path) -> tuple[list[dict], int, int]:
    """Group all official annotation records by physical image file name."""
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    polygons_by_image: dict[object, list[list[float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("The supplied annotations must be COCO polygons.")
        polygons_by_image[annotation["image_id"]].extend(segmentation)

    grouped = {}
    for image_record in document["images"]:
        file_name = str(image_record["file_name"])
        key = Path(file_name).stem
        observation = grouped.setdefault(
            key,
            {
                "observation_key": key,
                "file_name": file_name,
                "annotation_sets": [],
            },
        )
        if observation["file_name"] != file_name:
            raise ValueError(f"Observation key collision: {key}")
        observation["annotation_sets"].append(
            {
                "image_record_id": image_record["id"],
                "polygons": polygons_by_image.get(image_record["id"], []),
            }
        )
    return (
        [grouped[key] for key in sorted(grouped)],
        len(document["images"]),
        len(document["annotations"]),
    )


def assign_grouped_folds(
    observations: list[dict],
    *,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, int], str]:
    """Reproduce the immutable physical-observation folds."""
    keys = sorted(str(observation["observation_key"]) for observation in observations)
    random.Random(seed).shuffle(keys)
    assignments = {key: index % n_splits for index, key in enumerate(keys)}
    canonical = "\n".join(f"{key},{assignments[key]}" for key in sorted(assignments))
    return assignments, hashlib.sha256(canonical.encode()).hexdigest()


def rasterize_instances(
    polygons: list[list[float]],
    *,
    original_size: int,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one annotation set as semantic and integer instance masks."""
    if not polygons:
        shape = (model_size, model_size)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint16)
    scale = model_size / original_size
    scaled = [[coordinate * scale for coordinate in polygon] for polygon in polygons]
    decoded = mask_utils.decode(mask_utils.frPyObjects(scaled, model_size, model_size))
    if decoded.ndim == 2:
        decoded = decoded[:, :, np.newaxis]
    semantic = np.any(decoded != 0, axis=2).astype(np.uint8)
    instances = np.zeros((model_size, model_size), dtype=np.uint16)
    for index in range(decoded.shape[2]):
        instances[decoded[:, :, index] != 0] = index + 1
    return semantic, instances


def prepare_observations(
    observations: list[dict],
    *,
    image_directory: Path,
    original_size: int,
    model_size: int,
    disk_erosion_pixels: int,
) -> list[dict]:
    """Cache each grayscale image once with all official annotator targets."""
    prepared = []
    for index, observation in enumerate(observations, start=1):
        image_path = image_directory / Path(observation["file_name"]).name
        with Image.open(image_path) as image:
            resized = image.convert("L").resize(
                (model_size, model_size),
                Image.Resampling.BILINEAR,
            )
            image_array = np.asarray(resized, dtype=np.uint8)
        semantic_masks = []
        instance_masks = []
        for annotation_set in observation["annotation_sets"]:
            semantic, instances = rasterize_instances(
                annotation_set["polygons"],
                original_size=original_size,
                model_size=model_size,
            )
            semantic_masks.append(semantic)
            instance_masks.append(instances)
        consensus = np.mean(
            np.stack(semantic_masks),
            axis=0,
            dtype=np.float32,
        ).astype(np.float16)
        prepared.append(
            {
                **observation,
                "image": image_array,
                "disk_mask": solar_disk_mask(
                    image_array,
                    erosion_pixels=disk_erosion_pixels,
                ),
                "semantic_masks": semantic_masks,
                "instance_masks": instance_masks,
                "consensus": consensus,
                "annotator_count": len(semantic_masks),
            }
        )
        if index % 50 == 0 or index == len(observations):
            print(f"Prepared {index}/{len(observations)} physical observations.")
    return prepared


def build_model(base_channels: int):
    """Reconstruct the experiment-003 compact U-Net."""
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


def make_datasets():
    """Create training and validation Dataset classes lazily."""
    import torch

    class HybridDataset(torch.utils.data.Dataset):
        def __init__(self, samples, *, seed: int, augmentation_config: dict) -> None:
            self.samples = samples
            self.seed = seed
            self.epoch = 0
            self.augmentation_config = augmentation_config

        def __len__(self) -> int:
            return len(self.samples)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __getitem__(self, index: int):
            sample = self.samples[index]
            annotator_index = deterministic_annotator_index(
                self.seed,
                self.epoch,
                index,
                sample["annotator_count"],
            )
            image = sample["image"].copy()
            consensus = sample["consensus"].copy()
            annotator = sample["semantic_masks"][annotator_index].copy()
            if random.random() < self.augmentation_config["horizontal_flip_probability"]:
                image = np.fliplr(image)
                consensus = np.fliplr(consensus)
                annotator = np.fliplr(annotator)
            if random.random() < self.augmentation_config["vertical_flip_probability"]:
                image = np.flipud(image)
                consensus = np.flipud(consensus)
                annotator = np.flipud(annotator)
            if random.random() < self.augmentation_config["rotate_90_probability"]:
                turns = random.randint(1, 3)
                image = np.rot90(image, turns)
                consensus = np.rot90(consensus, turns)
                annotator = np.rot90(annotator, turns)
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
            return (
                torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0,
                torch.from_numpy(np.ascontiguousarray(consensus)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(annotator)).float().unsqueeze(0),
            )

    class ValidationDataset(torch.utils.data.Dataset):
        def __init__(self, samples) -> None:
            self.samples = [
                (sample["image"], mask) for sample in samples for mask in sample["semantic_masks"]
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            image, mask = self.samples[index]
            return (
                torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0,
                torch.from_numpy(mask.copy()).float().unsqueeze(0),
            )

    return HybridDataset, ValidationDataset


def base_loss(logits, targets, *, config: dict):
    """Compute fixed weighted BCE plus soft Dice."""
    import torch
    from torch.nn import functional

    positive_weight = torch.tensor(config["bce_positive_weight"], device=logits.device)
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return config["bce_loss_weight"] * bce + config["dice_loss_weight"] * dice_loss


def hybrid_loss(logits, consensus, annotator, *, training_config: dict, weights: dict):
    """Combine consensus and deterministic annotator-specific losses."""
    consensus_loss = base_loss(logits, consensus, config=training_config)
    annotator_loss = base_loss(logits, annotator, config=training_config)
    return weights["consensus"] * consensus_loss + weights["annotator"] * annotator_loss


def semantic_dice(probabilities, targets, *, threshold: float = 0.5) -> np.ndarray:
    """Return per-record semantic Dice."""
    predictions = probabilities >= threshold
    expected = targets >= 0.5
    intersection = (predictions & expected).sum(axis=(1, 2, 3))
    denominator = predictions.sum(axis=(1, 2, 3)) + expected.sum(axis=(1, 2, 3))
    return np.where(denominator == 0, 1.0, 2.0 * intersection / denominator)


def train_model(
    model,
    training_dataset,
    train_loader,
    validation_loader,
    *,
    config: dict,
    weights: dict,
    device,
):
    """Fine-tune one grouped fold and retain its best validation checkpoint."""
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
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        training_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for images, consensus, annotator in train_loader:
            images = images.to(device, non_blocking=True)
            consensus = consensus.to(device, non_blocking=True)
            annotator = annotator.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = hybrid_loss(
                    logits,
                    consensus,
                    annotator,
                    training_config=config["training"],
                    weights=weights,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        scores = []
        with torch.no_grad():
            for images, targets in validation_loader:
                probabilities = (
                    torch.sigmoid(model(images.to(device, non_blocking=True))).cpu().numpy()
                )
                scores.extend(semantic_dice(probabilities, targets.numpy()).tolist())
        mean_loss = float(np.mean(losses))
        mean_score = float(np.mean(scores))
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
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        scheduler.step()
        if (
            epoch >= config["training"]["minimum_epochs"]
            and stale_epochs >= config["training"]["early_stopping_patience"]
        ):
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
            break
    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_state, best_score, best_epoch, history


def predict_with_tta(model, inputs, *, augmentations: list[str]):
    """Average aligned probabilities over deterministic flips."""
    import torch

    outputs = []
    for augmentation in augmentations:
        if augmentation == "identity":
            transformed = inputs
            inverse: tuple[int, ...] = ()
        elif augmentation == "horizontal-flip":
            inverse = (-1,)
            transformed = torch.flip(inputs, inverse)
        elif augmentation == "vertical-flip":
            inverse = (-2,)
            transformed = torch.flip(inputs, inverse)
        elif augmentation == "horizontal-vertical-flip":
            inverse = (-2, -1)
            transformed = torch.flip(inputs, inverse)
        else:
            raise ValueError(f"Unsupported TTA: {augmentation}")
        probability = torch.sigmoid(model(transformed))
        outputs.append(torch.flip(probability, inverse) if inverse else probability)
    return torch.stack(outputs).mean(dim=0)


def predict_observations(model, observations: list[dict], *, config: dict, device) -> list:
    """Predict one probability map per physical held-out observation."""
    import torch

    predictions = []
    model.eval()
    with torch.no_grad():
        for index, observation in enumerate(observations, start=1):
            tensor = (
                torch.from_numpy(observation["image"].copy())
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
                / 255.0
            )
            probability = predict_with_tta(
                model,
                tensor,
                augmentations=config["inference"]["test_time_augmentations"],
            )[0, 0]
            predictions.append(probability.cpu().numpy())
            if index % 30 == 0 or index == len(observations):
                print(f"OOF inference {index}/{len(observations)} observations.")
    return predictions


def solar_disk_mask(image: np.ndarray, *, erosion_pixels: int) -> np.ndarray:
    """Estimate and erode the bright solar disk."""
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        thresholded,
        connectivity=8,
    )
    if count <= 1:
        return np.ones_like(image, dtype=bool)
    largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    disk = (labels == largest).astype(np.uint8)
    if erosion_pixels > 0:
        size = erosion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        disk = cv2.erode(disk, kernel)
    return disk != 0


def separate_instances(
    probability: np.ndarray,
    image: np.ndarray,
    *,
    config: dict,
    disk_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Apply the frozen experiment-003 OOF-selected instance pipeline."""
    foreground = probability >= config["probability_threshold"]
    if disk_mask is None:
        disk_mask = solar_disk_mask(
            image,
            erosion_pixels=config["disk_erosion_pixels"],
        )
    foreground &= disk_mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config["closing_kernel"], config["closing_kernel"]),
    )
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config["closing_iterations"],
    )
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=config["connectivity"],
    )
    candidates = []
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if not (
            config["min_component_area_at_model_resolution"]
            <= area
            <= config["max_component_area_at_model_resolution"]
        ):
            continue
        component = labels == label
        if float(probability[component].mean()) < config["minimum_component_mean_probability"]:
            continue
        centroid_x, centroid_y = centroids[label]
        candidates.append((-area, float(centroid_y), float(centroid_x), label))
    candidates.sort()
    return [(labels == label).astype(np.uint8) for _, _, _, label in candidates]


def instance_diagnostic(
    predictions: list[np.ndarray],
    targets: np.ndarray,
    *,
    minimum_iou: float,
) -> dict:
    """Greedily match instances and penalize missed and extra predictions."""
    target_ids = np.unique(targets)
    target_ids = target_ids[target_ids != 0]
    target_areas = {
        int(target_id): int(np.count_nonzero(targets == target_id)) for target_id in target_ids
    }
    pairs = []
    overlaps_by_prediction = []
    overlaps_by_target = {int(target_id): 0 for target_id in target_ids}
    for prediction_index, prediction in enumerate(predictions):
        prediction_area = int(prediction.sum())
        ids, intersections = np.unique(targets[prediction != 0], return_counts=True)
        overlap_count = 0
        for target_id, intersection in zip(ids, intersections, strict=True):
            target_id = int(target_id)
            if target_id == 0:
                continue
            intersection = int(intersection)
            union = prediction_area + target_areas[target_id] - intersection
            iou = intersection / union
            dice = 2.0 * intersection / (prediction_area + target_areas[target_id])
            if iou >= minimum_iou:
                pairs.append((dice, iou, prediction_index, target_id))
                overlap_count += 1
                overlaps_by_target[target_id] += 1
        overlaps_by_prediction.append(overlap_count)
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
    denominator = max(len(predictions), len(target_ids), 1)
    return {
        "penalized_dice": float(sum(matched_dice) / denominator),
        "matched_dice_sum": float(sum(matched_dice)),
        "matched_iou_sum": float(sum(matched_iou)),
        "matched": len(matched_dice),
        "missed": len(target_ids) - len(matched_targets),
        "extra": len(predictions) - len(matched_predictions),
        "one_to_many": sum(count > 1 for count in overlaps_by_target.values()),
        "many_to_one": sum(count > 1 for count in overlaps_by_prediction),
    }


def empty_aggregate() -> dict:
    """Create one diagnostic accumulator."""
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
        "per_image_penalized_dice": [],
    }


def add_diagnostic(aggregate: dict, diagnostic: dict) -> None:
    """Add one annotation-record result."""
    aggregate["images"] += 1
    aggregate["penalized_dice_sum"] += diagnostic["penalized_dice"]
    aggregate["per_image_penalized_dice"].append(diagnostic["penalized_dice"])
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


def merge_aggregate(destination: dict, source: dict) -> None:
    """Merge fold diagnostics."""
    for name, value in source.items():
        if name == "per_image_penalized_dice":
            destination[name].extend(value)
        else:
            destination[name] += value


def summarize_aggregate(aggregate: dict) -> dict:
    """Calculate stable instance means and distribution quantiles."""
    values = np.asarray(aggregate["per_image_penalized_dice"], dtype=np.float64)
    return {
        **{name: value for name, value in aggregate.items() if name != "per_image_penalized_dice"},
        "mean_penalized_instance_dice": aggregate["penalized_dice_sum"]
        / max(aggregate["images"], 1),
        "mean_matched_instance_dice": aggregate["matched_dice_sum"] / max(aggregate["matched"], 1),
        "mean_matched_instance_iou": aggregate["matched_iou_sum"] / max(aggregate["matched"], 1),
        "per_image_penalized_dice_quantiles": {
            "q10": float(np.quantile(values, 0.1)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.5)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.9)),
        },
    }


def summarize_totals(aggregate: dict) -> dict:
    """Summarize merged folds when per-record distributions are stored per fold."""
    return {
        "images": aggregate["images"],
        "mean_penalized_instance_dice": aggregate["penalized_dice_sum"]
        / max(aggregate["images"], 1),
        "mean_matched_instance_dice": aggregate["matched_dice_sum"] / max(aggregate["matched"], 1),
        "mean_matched_instance_iou": aggregate["matched_iou_sum"] / max(aggregate["matched"], 1),
        **{
            name: aggregate[name]
            for name in ("matched", "missed", "extra", "one_to_many", "many_to_one")
        },
    }


def evaluate_oof(
    observations: list[dict],
    probabilities: list[np.ndarray],
    *,
    config: dict,
) -> dict:
    """Evaluate fixed post-processing across annotator-count strata."""
    aggregate = empty_aggregate()
    by_annotator_count: dict[int, dict] = {}
    semantic_scores = []
    for observation, probability in zip(observations, probabilities, strict=True):
        predictions = separate_instances(
            probability,
            observation["image"],
            config=config["postprocessing"],
            disk_mask=observation["disk_mask"],
        )
        stratum = by_annotator_count.setdefault(
            observation["annotator_count"],
            empty_aggregate(),
        )
        for semantic, instances in zip(
            observation["semantic_masks"],
            observation["instance_masks"],
            strict=True,
        ):
            semantic_scores.append(
                float(
                    semantic_dice(
                        probability[np.newaxis, np.newaxis],
                        semantic[np.newaxis, np.newaxis],
                    )[0]
                )
            )
            diagnostic = instance_diagnostic(
                predictions,
                instances,
                minimum_iou=config["postprocessing"]["matching_min_iou"],
            )
            add_diagnostic(aggregate, diagnostic)
            add_diagnostic(stratum, diagnostic)
    return {
        "instance": summarize_aggregate(aggregate),
        "by_annotator_count": {
            str(count): summarize_aggregate(values)
            for count, values in sorted(by_annotator_count.items())
        },
        "mean_semantic_dice": float(np.mean(semantic_scores)),
    }


def weight_key(weights: dict) -> str:
    """Return a stable label for one fixed hybrid pair."""
    return f"consensus={weights['consensus']:.2f},annotator={weights['annotator']:.2f}"


def train_fold(
    prepared: list[dict],
    assignments: dict[str, int],
    fold_index: int,
    *,
    weights: dict,
    config: dict,
    device,
):
    """Fine-tune and evaluate one grouped fold."""
    import torch

    fold_seed = config["experiment"]["seed"] + fold_index
    seed_everything(fold_seed)
    training = [
        sample for sample in prepared if assignments[sample["observation_key"]] != fold_index
    ]
    validation = [
        sample for sample in prepared if assignments[sample["observation_key"]] == fold_index
    ]
    hybrid_dataset_class, validation_dataset_class = make_datasets()
    training_dataset = hybrid_dataset_class(
        training,
        seed=fold_seed,
        augmentation_config=config["training"]["augmentations"],
    )
    validation_dataset = validation_dataset_class(validation)
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
    checkpoint = resolve_checkpoint(config, "experiment-003", fold_index)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    print(
        f"Fold {fold_index}, {weight_key(weights)}: "
        f"train observations={len(training)}, validation observations={len(validation)}, "
        f"source={checkpoint.name}"
    )
    model, state, best_score, best_epoch, history = train_model(
        model,
        training_dataset,
        train_loader,
        validation_loader,
        config=config,
        weights=weights,
        device=device,
    )
    probabilities = predict_observations(model, validation, config=config, device=device)
    evaluation = evaluate_oof(validation, probabilities, config=config)
    record = {
        "fold": fold_index,
        "seed": fold_seed,
        "weights": weights,
        "training_observations": len(training),
        "validation_observations": len(validation),
        "validation_annotation_sets": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_semantic_dice": best_score,
        "tta_oof_evaluation": evaluation,
        "source_checkpoint": checkpoint.name,
        "training_history": history,
    }
    del model, train_loader, validation_loader
    torch.cuda.empty_cache()
    return state, record


def screen_weights(prepared: list[dict], assignments: dict, *, config: dict, device):
    """Screen both fixed pairs on folds 0 and 1 using OOF instance Dice."""
    candidates = {}
    for weights in config["screen"]["weight_pairs"]:
        key = weight_key(weights)
        aggregate = empty_aggregate()
        states = {}
        fold_records = []
        for fold_index in config["screen"]["folds"]:
            state, record = train_fold(
                prepared,
                assignments,
                fold_index,
                weights=weights,
                config=config,
                device=device,
            )
            states[fold_index] = state
            fold_records.append(record)
            fold_instance = record["tta_oof_evaluation"]["instance"]
            # Reconstructing only aggregate sums is sufficient for screen selection.
            for name in aggregate:
                if name != "per_image_penalized_dice" and name in fold_instance:
                    aggregate[name] += fold_instance[name]
        score = aggregate["penalized_dice_sum"] / max(aggregate["images"], 1)
        candidates[key] = {
            "weights": weights,
            "states": states,
            "folds": fold_records,
            "mean_penalized_instance_dice": score,
        }
        print(f"Screen candidate {key}: penalized_instance_dice={score:.6f}")
    selected_key = max(
        candidates,
        key=lambda key: (
            candidates[key]["mean_penalized_instance_dice"],
            candidates[key]["weights"]["consensus"],
        ),
    )
    print(f"Selected hybrid screen candidate: {selected_key}")
    return candidates, selected_key


def promotion_result(summary: dict, *, config: dict) -> dict:
    """Evaluate the predeclared full-five-fold promotion gate."""
    promotion = config["promotion"]
    checks = {
        "mean_penalized_instance_dice": (
            summary["mean_penalized_instance_dice"]
            >= promotion["minimum_mean_penalized_instance_dice"]
        ),
        "missed_instances": (
            not promotion["require_lower_missed_instances"]
            or summary["missed"] < promotion["baseline_missed_instances"]
        ),
        "extra_instances": summary["extra"] <= promotion["maximum_extra_instances"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode one non-empty fixed-size instance."""
    if not np.any(mask):
        raise ValueError("Cannot encode an empty instance.")
    encoded = mask_utils.encode(np.asfortranarray((mask != 0).astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str) or not counts:
        raise ValueError("Invalid COCO compressed RLE.")
    return counts


def prepare_test_images(test_directory: Path, *, model_size: int) -> dict[str, np.ndarray]:
    """Load test observations only for final inference."""
    paths = sorted(test_directory.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No test JPEGs in {test_directory}.")
    prepared = {}
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("L").resize(
                (model_size, model_size),
                Image.Resampling.BILINEAR,
            )
            prepared[path.stem] = np.asarray(resized, dtype=np.uint8)
    return prepared


def ensemble_test_probabilities(
    states: dict[int, dict], test_images: dict, *, config: dict, device
):
    """Average all five selected hybrid models on final test observations."""
    import torch

    sums = {
        image_id: np.zeros_like(image, dtype=np.float32) for image_id, image in test_images.items()
    }
    for fold_index in config["validation"]["fold_indices"]:
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(states[fold_index])
        model.eval()
        with torch.no_grad():
            for index, (image_id, image) in enumerate(test_images.items(), start=1):
                tensor = (
                    torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                    / 255.0
                )
                probability = predict_with_tta(
                    model,
                    tensor,
                    augmentations=config["inference"]["test_time_augmentations"],
                )[0, 0]
                sums[image_id] += probability.cpu().numpy()
                if index % 30 == 0 or index == len(test_images):
                    print(f"Fold {fold_index} test inference {index}/{len(test_images)}.")
        del model
        torch.cuda.empty_cache()
    return {
        image_id: probability_sum / len(config["validation"]["fold_indices"])
        for image_id, probability_sum in sums.items()
    }


def submission_rows(probabilities: dict, test_images: dict, *, config: dict):
    """Separate and encode final ensemble instances."""
    rows = []
    counts_by_image = {}
    for index, (image_id, image) in enumerate(test_images.items(), start=1):
        instances = separate_instances(
            probabilities[image_id],
            image,
            config=config["postprocessing"],
        )
        seen = set()
        for instance_index, instance in enumerate(instances, start=1):
            resized = cv2.resize(
                instance,
                (config["submission"]["mask_width"], config["submission"]["mask_height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            counts = encode_binary_mask(resized)
            if counts in seen:
                raise ValueError(f"Duplicate mask for {image_id}.")
            seen.add(counts)
            rows.append((f"{image_id}_{instance_index}", counts))
        counts_by_image[image_id] = len(instances)
        print(f"Final output {index}/{len(test_images)}: {image_id}, instances={len(instances)}")
    return rows, counts_by_image


def validate_and_write(rows: list[tuple[str, str]], *, known_ids: set[str], path: Path) -> None:
    """Validate identifiers and canonical RLE before writing."""
    if not rows:
        raise ValueError("Submission must contain at least one instance.")
    seen_ids = set()
    masks_by_image: dict[str, set[str]] = defaultdict(set)
    for filament_id, counts in rows:
        image_id, separator, suffix = filament_id.rpartition("_")
        if not separator or image_id not in known_ids or not suffix.isdigit():
            raise ValueError(f"Invalid filament_id: {filament_id}")
        if filament_id in seen_ids or counts in masks_by_image[image_id]:
            raise ValueError(f"Duplicate identifier or mask: {filament_id}")
        seen_ids.add(filament_id)
        masks_by_image[image_id].add(counts)
        decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("ascii")})
        if not np.any(decoded) or encode_binary_mask(decoded) != counts:
            raise ValueError(f"Invalid RLE round trip: {filament_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)


def calibration_candidates(config: dict) -> list[dict]:
    """Build the frozen OOF-only blend and post-processing grid."""
    calibration = config["calibration"]
    candidates = []
    for experiment_005_weight in calibration["experiment_005_probability_weights"]:
        for threshold in calibration["probability_thresholds"]:
            for closing_kernel in calibration["closing_kernels"]:
                for minimum_area in calibration["minimum_component_areas"]:
                    for minimum_confidence in calibration["minimum_component_mean_probabilities"]:
                        postprocessing = dict(config["postprocessing"])
                        postprocessing.update(
                            {
                                "probability_threshold": float(threshold),
                                "closing_kernel": int(closing_kernel),
                                "min_component_area_at_model_resolution": int(minimum_area),
                                "minimum_component_mean_probability": float(minimum_confidence),
                            }
                        )
                        candidates.append(
                            {
                                "experiment_005_probability_weight": float(experiment_005_weight),
                                "postprocessing": postprocessing,
                            }
                        )
    return candidates


def blended_probabilities(
    experiment_003: list[np.ndarray],
    experiment_005: list[np.ndarray],
    *,
    experiment_005_weight: float,
):
    """Yield aligned OOF probability blends without retaining another full copy."""
    if not 0.0 <= experiment_005_weight <= 1.0:
        raise ValueError("Blend weight must be in [0, 1].")
    for probability_003, probability_005 in zip(
        experiment_003,
        experiment_005,
        strict=True,
    ):
        if experiment_005_weight == 0.0:
            yield probability_003
        elif experiment_005_weight == 1.0:
            yield probability_005
        else:
            yield (
                (1.0 - experiment_005_weight) * probability_003
                + experiment_005_weight * probability_005
            )


def evaluate_instance_oof(
    observations: list[dict],
    probabilities,
    *,
    postprocessing: dict,
) -> dict:
    """Evaluate instance diagnostics without repeating semantic-mask work."""
    aggregate = empty_aggregate()
    by_annotator_count: dict[int, dict] = {}
    for observation, probability in zip(observations, probabilities, strict=True):
        predictions = separate_instances(
            probability,
            observation["image"],
            config=postprocessing,
            disk_mask=observation["disk_mask"],
        )
        stratum = by_annotator_count.setdefault(
            observation["annotator_count"],
            empty_aggregate(),
        )
        for instances in observation["instance_masks"]:
            diagnostic = instance_diagnostic(
                predictions,
                instances,
                minimum_iou=postprocessing["matching_min_iou"],
            )
            add_diagnostic(aggregate, diagnostic)
            add_diagnostic(stratum, diagnostic)
    return {
        "instance": summarize_aggregate(aggregate),
        "by_annotator_count": {
            str(count): summarize_aggregate(values)
            for count, values in sorted(by_annotator_count.items())
        },
    }


def candidate_matches(candidate: dict, reference: dict) -> bool:
    """Return whether a candidate has the five reference-defining values."""
    postprocessing = candidate["postprocessing"]
    return (
        np.isclose(
            candidate["experiment_005_probability_weight"],
            reference["experiment_005_probability_weight"],
        )
        and np.isclose(
            postprocessing["probability_threshold"],
            reference["probability_threshold"],
        )
        and postprocessing["closing_kernel"] == reference["closing_kernel"]
        and postprocessing["min_component_area_at_model_resolution"]
        == reference["min_component_area_at_model_resolution"]
        and np.isclose(
            postprocessing["minimum_component_mean_probability"],
            reference["minimum_component_mean_probability"],
        )
    )


def validate_reference_candidates(results: list[dict], *, config: dict) -> dict:
    """Prove that both checkpoint families reproduce their recorded OOF results."""
    reference_config = config["calibration"]["reference_candidates"]
    tolerance = reference_config["dice_absolute_tolerance"]
    checks = {}
    for source in ("experiment-003", "experiment-005"):
        expected = reference_config[source]
        matches = [result for result in results if candidate_matches(result, expected)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {source} reference candidate, got {len(matches)}."
            )
        summary = matches[0]["instance"]
        source_checks = {
            "mean_penalized_instance_dice": abs(
                summary["mean_penalized_instance_dice"]
                - expected["expected_mean_penalized_instance_dice"]
            )
            <= tolerance,
            "missed_instances": summary["missed"] == expected["expected_missed_instances"],
            "extra_instances": summary["extra"] == expected["expected_extra_instances"],
        }
        checks[source] = {
            "passed": all(source_checks.values()),
            "checks": source_checks,
            "observed": {
                "mean_penalized_instance_dice": summary["mean_penalized_instance_dice"],
                "missed": summary["missed"],
                "extra": summary["extra"],
            },
        }
    if not all(record["passed"] for record in checks.values()):
        raise RuntimeError(f"Reference OOF regression failed: {checks}")
    return checks


def select_calibration_candidate(results: list[dict], *, config: dict) -> dict:
    """Select by Dice, preferring the predeclared missed/extra feasible set."""
    feasibility = config["calibration"]["feasibility"]
    for result in results:
        summary = result["instance"]
        result["feasible"] = (
            summary["missed"] <= feasibility["maximum_missed_instances"]
            and summary["extra"] <= feasibility["maximum_extra_instances"]
        )
    pool = [result for result in results if result["feasible"]]
    if not pool:
        pool = results
    return max(
        enumerate(pool),
        key=lambda indexed: (
            indexed[1]["instance"]["mean_penalized_instance_dice"],
            indexed[1]["instance"]["mean_matched_instance_dice"],
            -indexed[1]["instance"]["extra"],
            -indexed[1]["instance"]["missed"],
            -indexed[0],
        ),
    )[1]


def blend_promotion_result(summary: dict, *, config: dict) -> dict:
    """Evaluate the predeclared experiment-006 promotion gate."""
    promotion = config["promotion"]
    checks = {
        "mean_penalized_instance_dice": summary["mean_penalized_instance_dice"]
        >= promotion["minimum_mean_penalized_instance_dice"],
        "missed_instances": summary["missed"] <= promotion["maximum_missed_instances"],
        "extra_instances": summary["extra"] <= promotion["maximum_extra_instances"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def predict_grouped_oof(
    prepared: list[dict],
    assignments: dict[str, int],
    *,
    source: str,
    config: dict,
    device,
) -> list[np.ndarray]:
    """Reconstruct one OOF map per physical observation from fold checkpoints."""
    import torch

    by_key = {}
    for fold_index in config["validation"]["fold_indices"]:
        validation = [
            observation
            for observation in prepared
            if assignments[observation["observation_key"]] == fold_index
        ]
        checkpoint = resolve_checkpoint(config, source, fold_index)
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(
            f"{source} fold {fold_index}: OOF observations={len(validation)}, "
            f"checkpoint={checkpoint}"
        )
        probabilities = predict_observations(
            model,
            validation,
            config=config,
            device=device,
        )
        for observation, probability in zip(
            validation,
            probabilities,
            strict=True,
        ):
            key = observation["observation_key"]
            if key in by_key:
                raise RuntimeError(f"Duplicate OOF prediction: {key}")
            by_key[key] = probability
        del model
        torch.cuda.empty_cache()
    if set(by_key) != {observation["observation_key"] for observation in prepared}:
        raise RuntimeError(f"Incomplete OOF coverage for {source}.")
    return [by_key[observation["observation_key"]] for observation in prepared]


def ensemble_source_test_probabilities(
    source: str,
    test_images: dict[str, np.ndarray],
    *,
    config: dict,
    device,
) -> dict[str, np.ndarray]:
    """Average all five fold checkpoints for one approved source."""
    import torch

    sums = {
        image_id: np.zeros_like(image, dtype=np.float32) for image_id, image in test_images.items()
    }
    folds = config["validation"]["fold_indices"]
    for fold_index in folds:
        checkpoint = resolve_checkpoint(config, source, fold_index)
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            for index, (image_id, image) in enumerate(test_images.items(), start=1):
                tensor = (
                    torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                    / 255.0
                )
                probability = predict_with_tta(
                    model,
                    tensor,
                    augmentations=config["inference"]["test_time_augmentations"],
                )[0, 0]
                sums[image_id] += probability.cpu().numpy()
                if index % 30 == 0 or index == len(test_images):
                    print(f"{source} fold {fold_index} test inference {index}/{len(test_images)}.")
        del model
        torch.cuda.empty_cache()
    return {image_id: probability_sum / len(folds) for image_id, probability_sum in sums.items()}


def main() -> None:
    """Calibrate two checkpoint families on grouped OOF and infer test data."""
    import torch

    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    if config["experiment"]["mode"] != "checkpoint-blend-and-oof-calibration":
        raise ValueError("Experiment 006 must remain an OOF calibration run.")
    if config["calibration"]["selection_metric"] != "mean-penalized-instance-dice":
        raise ValueError("Calibration must select by OOF penalized instance Dice.")
    if config["model"]["competition_trained_checkpoint_sources"] != [
        "experiment-003",
        "experiment-005",
    ]:
        raise ValueError("Experiment 006 requires only experiments 003 and 005.")
    if config["model"]["pretrained_weights"] or config["model"]["external_labeled_data"]:
        raise ValueError("External supervision is disabled for experiment 006.")
    seed_everything(config["experiment"]["seed"])
    data_root = resolve_data_root(config)
    observations, annotation_sets, annotations = load_observation_records(
        data_root / config["data"]["train_annotations"]
    )
    assignments, fingerprint = assign_grouped_folds(
        observations,
        n_splits=config["validation"]["n_splits"],
        seed=config["validation"]["seed"],
    )
    if fingerprint != config["validation"]["expected_fold_fingerprint"]:
        raise RuntimeError(f"Fold fingerprint changed: {fingerprint}")
    prepared = prepare_observations(
        observations,
        image_directory=data_root / config["data"]["train_images"],
        original_size=config["competition"]["image_height"],
        model_size=config["model"]["input_size"],
        disk_erosion_pixels=config["postprocessing"]["disk_erosion_pixels"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_hash = hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest()
    print(f"Official data root: {data_root}")
    print(f"Config SHA-256: {config_hash}")
    print(f"Fold fingerprint: {fingerprint}")
    print(f"Inference device: {device}")

    oof_003 = predict_grouped_oof(
        prepared,
        assignments,
        source="experiment-003",
        config=config,
        device=device,
    )
    oof_005 = predict_grouped_oof(
        prepared,
        assignments,
        source="experiment-005",
        config=config,
        device=device,
    )
    candidates = calibration_candidates(config)
    candidate_results = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = evaluate_instance_oof(
            prepared,
            blended_probabilities(
                oof_003,
                oof_005,
                experiment_005_weight=candidate["experiment_005_probability_weight"],
            ),
            postprocessing=candidate["postprocessing"],
        )
        record = {**candidate, **evaluation}
        candidate_results.append(record)
        summary = evaluation["instance"]
        print(
            f"Calibration {index}/{len(candidates)}: "
            f"w005={candidate['experiment_005_probability_weight']:.3f}, "
            f"threshold={candidate['postprocessing']['probability_threshold']:.3f}, "
            f"area={candidate['postprocessing']['min_component_area_at_model_resolution']}, "
            f"confidence={candidate['postprocessing']['minimum_component_mean_probability']:.3f}, "
            f"dice={summary['mean_penalized_instance_dice']:.6f}, "
            f"missed={summary['missed']}, extra={summary['extra']}"
        )
    reference_checks = validate_reference_candidates(
        candidate_results,
        config=config,
    )
    selected = select_calibration_candidate(candidate_results, config=config)
    selected_config = {**config, "postprocessing": selected["postprocessing"]}
    selected_evaluation = evaluate_oof(
        prepared,
        blended_probabilities(
            oof_003,
            oof_005,
            experiment_005_weight=selected["experiment_005_probability_weight"],
        ),
        config=selected_config,
    )
    selected_summary = selected_evaluation["instance"]
    promotion = blend_promotion_result(selected_summary, config=config)
    selected_candidate = {
        "experiment_005_probability_weight": selected["experiment_005_probability_weight"],
        "postprocessing": selected["postprocessing"],
        "feasible": selected["feasible"],
    }
    oof_metadata = {
        "experiment_id": config["experiment"]["id"],
        "config_sha256": config_hash,
        "fold_fingerprint": fingerprint,
        "checkpoint_sources": config["model"]["competition_trained_checkpoint_sources"],
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "reference_checks": reference_checks,
        "selected_candidate": selected_candidate,
        "full_oof": selected_summary,
        "full_oof_by_annotator_count": selected_evaluation["by_annotator_count"],
        "mean_semantic_dice": selected_evaluation["mean_semantic_dice"],
        "promotion_gate": promotion,
        "external_labeled_data": config["model"]["external_labeled_data"],
        "pretrained_weights": config["model"]["pretrained_weights"],
        "test_statistics_used_for_selection": False,
        "physical_observations": len(prepared),
        "annotation_sets": annotation_sets,
        "annotations": annotations,
        "oof_runtime_seconds": time.time() - started,
    }
    oof_path = Path("/kaggle/working/experiment-006-oof-metadata.json")
    oof_path.write_text(json.dumps(oof_metadata, indent=2), encoding="utf-8")
    print(
        "Selected full OOF: "
        f"dice={selected_summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={selected_summary['missed']}, "
        f"extra={selected_summary['extra']}, "
        f"promotion_gate_passed={promotion['passed']}"
    )
    print(f"OOF evidence written before test inference: {oof_path}")
    del oof_003, oof_005, candidate_results

    test_images = prepare_test_images(
        data_root / config["data"]["test_images"],
        model_size=config["model"]["input_size"],
    )
    probabilities_003 = ensemble_source_test_probabilities(
        "experiment-003",
        test_images,
        config=config,
        device=device,
    )
    probabilities_005 = ensemble_source_test_probabilities(
        "experiment-005",
        test_images,
        config=config,
        device=device,
    )
    weight_005 = selected_candidate["experiment_005_probability_weight"]
    blended_test = {
        image_id: (1.0 - weight_005) * probabilities_003[image_id]
        + weight_005 * probabilities_005[image_id]
        for image_id in test_images
    }
    del probabilities_003, probabilities_005
    rows, counts_by_image = submission_rows(
        blended_test,
        test_images,
        config=selected_config,
    )
    submission_path = Path(config["submission"]["output_path"])
    validate_and_write(rows, known_ids=set(test_images), path=submission_path)
    final_metadata = {
        **oof_metadata,
        "test_images": len(test_images),
        "submission_rows": len(rows),
        "predicted_instances_per_image": counts_by_image,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-006-run-metadata.json")
    metadata_path.write_text(
        json.dumps(final_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Candidate written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This private kernel creates artifacts only; it does not submit.")


if __name__ == "__main__":
    main()
