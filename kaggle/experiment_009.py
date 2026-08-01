"""Private Kaggle GPU script for experiment-009 low-learning-rate continuation.

The selected 1024-pixel, 32-channel models from experiment-008 are continued
for a fixed low-learning-rate budget on all five immutable grouped folds.
Targets, augmentations, TTA, post-processing, and test separation remain
unchanged so the experiment isolates whether the prior models were
under-optimized.
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
  id: experiment-009
  name: low-lr-continuation
  seed: 20260804
  output_root: outputs/experiments
  mode: five-fold-low-lr-continuation

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
  checkpoint_root: /kaggle/input/solar-filament-capacity-resolution-ablation

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
  architecture: small-unet-low-lr-continuation
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 32
  pretrained_weights: false
  external_labeled_data: false
  initialization: experiment-008-selected
  checkpoint_experiment: experiment-008

training:
  target_strategy: hybrid-consensus-random-annotator
  consensus_weight: 0.75
  annotator_weight: 0.25
  epochs: 14
  minimum_epochs: 6
  early_stopping_patience: 3
  batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 0.00005
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
  batch_size: 1
  test_time_augmentations:
    - identity
    - horizontal-flip
    - vertical-flip
    - horizontal-vertical-flip

postprocessing:
  source: experiment-006-selected
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
  baseline_experiment: experiment-008
  baseline_mean_penalized_instance_dice: 0.4957709510589161
  baseline_missed_instances: 1669
  baseline_extra_instances: 1935
  minimum_mean_penalized_instance_dice: 0.498
  maximum_missed_instances: 1669
  maximum_extra_instances: 1935

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

submission:
  columns:
    - filament_id
    - segmentation_rle
  rle_format: coco-compressed-counts
  mask_height: 2048
  mask_width: 2048
  output_path: /kaggle/working/experiment-009-submission.csv
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


def resolve_checkpoint(config: dict, fold_index: int) -> Path:
    """Resolve exactly one selected experiment-008 fold checkpoint."""
    file_name = f"experiment-008-fold-{fold_index}.pt"
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
        accumulation_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, consensus, annotator) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            consensus = consensus.to(device, non_blocking=True)
            annotator = annotator.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = hybrid_loss(
                    logits,
                    consensus,
                    annotator,
                    training_config=config["training"],
                    weights=weights,
                )
            scaler.scale(loss / accumulation_steps).backward()
            is_last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % accumulation_steps == 0 or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
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
) -> list[np.ndarray]:
    """Apply the frozen experiment-003 OOF-selected instance pipeline."""
    foreground = probability >= config["probability_threshold"]
    foreground &= solar_disk_mask(image, erosion_pixels=config["disk_erosion_pixels"])
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


def merge_evaluation_totals(destination: dict, evaluation: dict) -> None:
    """Merge exact additive OOF totals without rebuilding per-image masks."""
    summary = evaluation["instance"]
    for name in destination:
        if name != "per_image_penalized_dice" and name in summary:
            destination[name] += summary[name]


def screen_semantic_gate(record: dict, *, config: dict) -> bool:
    """Require the predeclared semantic improvement on every screen fold."""
    minimum_gain = config["screen"]["minimum_semantic_gain_on_each_screen_fold"]
    baselines = config["screen"]["baseline_screen_semantic_dice"]
    return all(
        fold["tta_oof_evaluation"]["mean_semantic_dice"]
        >= float(baselines[str(fold["fold"])]) + minimum_gain
        for fold in record["folds"]
    )


def select_setting(screen_results: list[dict], *, config: dict) -> tuple[dict, str]:
    """Select the best screen setting, preferring settings that pass the semantic gate."""
    if not screen_results:
        raise ValueError("At least one capacity/resolution setting is required.")
    gated = [result for result in screen_results if result["semantic_gate_passed"]]
    pool = gated if gated else screen_results
    selected = max(
        pool,
        key=lambda result: (
            result["mean_penalized_instance_dice"],
            result["mean_semantic_dice"],
            -result["setting"]["input_size"],
        ),
    )
    return selected["setting"], selected["setting"]["id"]


def train_fold(
    prepared: list[dict],
    assignments: dict[str, int],
    fold_index: int,
    *,
    setting: dict,
    config: dict,
    device,
):
    """Continue one selected experiment-008 checkpoint on one grouped fold."""
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
    model = build_model(setting["base_channels"]).to(device)
    checkpoint = resolve_checkpoint(config, fold_index)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    print(
        f"Fold {fold_index}, setting={setting['id']}: "
        f"train observations={len(training)}, validation observations={len(validation)}, "
        f"initialization={config['model']['initialization']}, source={checkpoint.name}"
    )
    model, state, best_score, best_epoch, history = train_model(
        model,
        training_dataset,
        train_loader,
        validation_loader,
        config=config,
        weights={
            "consensus": config["training"]["consensus_weight"],
            "annotator": config["training"]["annotator_weight"],
        },
        device=device,
    )
    probabilities = predict_observations(model, validation, config=config, device=device)
    evaluation = evaluate_oof(validation, probabilities, config=config)
    record = {
        "fold": fold_index,
        "seed": fold_seed,
        "setting": setting,
        "training_observations": len(training),
        "validation_observations": len(validation),
        "validation_annotation_sets": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_semantic_dice": best_score,
        "tta_oof_evaluation": evaluation,
        "initialization": config["model"]["initialization"],
        "source_checkpoint": checkpoint.name,
        "training_history": history,
    }
    del model, train_loader, validation_loader
    torch.cuda.empty_cache()
    return state, record, validation, probabilities


def promotion_result(summary: dict, *, config: dict) -> dict:
    """Evaluate the predeclared full-five-fold promotion gate."""
    promotion = config["promotion"]
    checks = {
        "mean_penalized_instance_dice": (
            summary["mean_penalized_instance_dice"]
            >= promotion["minimum_mean_penalized_instance_dice"]
        ),
        "missed_instances": summary["missed"] <= promotion["maximum_missed_instances"],
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
    states: dict[int, dict],
    test_images: dict,
    *,
    setting: dict,
    config: dict,
    device,
):
    """Average all five selected capacity/resolution models on final test observations."""
    import torch

    sums = {
        image_id: np.zeros_like(image, dtype=np.float32) for image_id, image in test_images.items()
    }
    for fold_index in config["validation"]["fold_indices"]:
        model = build_model(setting["base_channels"]).to(device)
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


def main() -> None:
    """Continue all five selected models, then perform final inference."""
    import torch

    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    if config["experiment"]["id"] != "experiment-009":
        raise ValueError("Embedded configuration must identify experiment-009.")
    if config["model"]["initialization"] != "experiment-008-selected":
        raise ValueError("Experiment-009 must initialize from experiment-008 checkpoints.")
    if config["model"]["checkpoint_experiment"] != "experiment-008":
        raise ValueError("Only experiment-008 checkpoints are allowed.")
    if config["model"]["pretrained_weights"] or config["model"]["external_labeled_data"]:
        raise ValueError("External or supervised pretrained data is forbidden.")
    if not np.isclose(
        config["training"]["consensus_weight"] + config["training"]["annotator_weight"],
        1.0,
    ):
        raise ValueError("Hybrid target weights must sum to one.")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_hash = hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest()
    setting = {
        "id": "1024-base32",
        "input_size": config["model"]["input_size"],
        "base_channels": config["model"]["base_channels"],
    }
    prepared = prepare_observations(
        observations,
        image_directory=data_root / config["data"]["train_images"],
        original_size=config["competition"]["image_height"],
        model_size=setting["input_size"],
    )
    print(f"Official data root: {data_root}")
    print(f"Config SHA-256: {config_hash}")
    print(f"Fold fingerprint: {fingerprint}")
    print(f"Training device: {device}")

    states = {}
    probabilities_by_key = {}
    fold_records = []
    for fold_index in config["validation"]["fold_indices"]:
        state, record, validation, probabilities = train_fold(
            prepared,
            assignments,
            fold_index,
            setting=setting,
            config=config,
            device=device,
        )
        states[fold_index] = state
        fold_records.append(record)
        for observation, probability in zip(validation, probabilities, strict=True):
            probabilities_by_key[observation["observation_key"]] = probability

    ordered_probabilities = [
        probabilities_by_key[observation["observation_key"]] for observation in prepared
    ]
    full_evaluation = evaluate_oof(prepared, ordered_probabilities, config=config)
    full_summary = full_evaluation["instance"]
    promotion = promotion_result(full_summary, config=config)
    if sorted(states) != config["validation"]["fold_indices"]:
        raise RuntimeError("Continuation states do not cover all five grouped folds.")
    oof_metadata = {
        "experiment_id": config["experiment"]["id"],
        "config_sha256": config_hash,
        "fold_fingerprint": fingerprint,
        "selected_setting": setting,
        "checkpoint_experiment": config["model"]["checkpoint_experiment"],
        "folds": sorted(fold_records, key=lambda record: record["fold"]),
        "full_oof": full_summary,
        "full_oof_by_annotator_count": full_evaluation["by_annotator_count"],
        "mean_semantic_dice": full_evaluation["mean_semantic_dice"],
        "promotion_gate": promotion,
        "initialization": config["model"]["initialization"],
        "external_labeled_data": config["model"]["external_labeled_data"],
        "pretrained_weights": config["model"]["pretrained_weights"],
        "test_statistics_used_for_selection": False,
    }
    oof_path = Path("/kaggle/working/experiment-009-oof-metadata.json")
    oof_path.write_text(json.dumps(oof_metadata, indent=2), encoding="utf-8")
    print(
        "Full OOF: "
        f"semantic_dice={full_evaluation['mean_semantic_dice']:.6f}, "
        f"penalized_dice={full_summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={full_summary['missed']}, extra={full_summary['extra']}, "
        f"promotion_gate_passed={promotion['passed']}"
    )
    print(f"OOF metadata checkpoint written before test inference: {oof_path}")

    for fold_index, state in states.items():
        torch.save(state, Path(f"/kaggle/working/experiment-009-fold-{fold_index}.pt"))
    test_images = prepare_test_images(
        data_root / config["data"]["test_images"],
        model_size=setting["input_size"],
    )
    probabilities = ensemble_test_probabilities(
        states,
        test_images,
        setting=setting,
        config=config,
        device=device,
    )
    rows, counts_by_image = submission_rows(probabilities, test_images, config=config)
    submission_path = Path(config["submission"]["output_path"])
    validate_and_write(rows, known_ids=set(test_images), path=submission_path)
    final_metadata = {
        **oof_metadata,
        "physical_observations": len(prepared),
        "annotation_sets": annotation_sets,
        "annotations": annotations,
        "test_images": len(test_images),
        "submission_rows": len(rows),
        "predicted_instances_per_image": counts_by_image,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-009-run-metadata.json")
    metadata_path.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"Candidate written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This private kernel creates artifacts only; it does not submit.")


if __name__ == "__main__":
    main()
