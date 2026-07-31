"""Private Kaggle GPU script for experiment-007 boundary-aware segmentation.

The script fine-tunes three-head U-Nets on the immutable grouped folds, learns
foreground, per-instance boundary, and normalized interior-distance targets,
selects connected-components or seeded partitioning using only grouped OOF
instance diagnostics, and loads test observations only after selection.
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
  id: experiment-007
  name: boundary-seeded-unet
  seed: 20260802
  output_root: outputs/experiments
  mode: boundary-aware-multitask-finetuning

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

model:
  task: instance-segmentation
  architecture: small-unet-three-head
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 24
  heads:
    - foreground
    - boundary
    - normalized-instance-distance
  pretrained_weights: false
  external_labeled_data: false
  competition_trained_checkpoint_source: experiment-003

targets:
  foreground: hybrid-consensus-random-annotator
  consensus_weight: 0.75
  annotator_weight: 0.25
  boundary_kernel: 3
  distance_normalization: per-instance-maximum
  deterministic_annotator_per_epoch: true

training:
  epochs: 16
  minimum_epochs: 8
  early_stopping_patience: 4
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.0001
  foreground_bce_positive_weight: 4.0
  boundary_bce_positive_weight: 8.0
  dice_loss_weight: 1.0
  bce_loss_weight: 1.0
  distance_foreground_weight: 4.0
  loss_weights:
    foreground: 1.0
    boundary: 0.5
    normalized_instance_distance: 0.25
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
  foreground_probability_threshold: 0.5
  closing_kernel: 7
  closing_iterations: 1
  disk_erosion_pixels: 8
  max_component_area_at_model_resolution: 40000
  connectivity: 8
  matching_min_iou: 0.1
  connected_components_baseline:
    min_component_area_at_model_resolution: 96
    minimum_component_mean_probability: 0.8
  seeded_partition_grid:
    boundary_probability_thresholds:
      - 0.4
      - 0.5
      - 0.6
    distance_seed_thresholds:
      - 0.35
      - 0.5
    seed_closing_kernels:
      - 5
      - 9
    minimum_seed_areas:
      - 8
      - 16
    minimum_instance_areas:
      - 64
      - 96
    minimum_instance_mean_probabilities:
      - 0.75
      - 0.8

selection:
  metric: mean-penalized-instance-dice
  prefer_feasible_candidates: true
  maximum_missed_instances: 1681
  maximum_extra_instances: 2095

promotion:
  baseline_experiment: experiment-006
  baseline_public_score: 0.66
  baseline_mean_penalized_instance_dice: 0.4917500437953408
  baseline_missed_instances: 1681
  baseline_extra_instances: 2095
  baseline_one_to_many: 514
  baseline_many_to_one: 120
  required_method: seeded-partition
  minimum_mean_penalized_instance_dice: 0.5
  maximum_missed_instances: 1681
  maximum_extra_instances: 2095
  maximum_one_to_many: 514
  maximum_many_to_one: 120

metric:
  primary: mean-matched-instance-dice
  diagnostics:
    - semantic-dice-distribution
    - matched-instance-dice
    - matched-instance-iou
    - missed-and-extra-instances
    - one-to-many-relations
    - many-to-one-relations
    - connected-components-versus-seeded-partition
    - single-versus-multi-annotator-dice

submission:
  columns:
    - filament_id
    - segmentation_rle
  rle_format: coco-compressed-counts
  mask_height: 2048
  mask_width: 2048
  output_path: /kaggle/working/experiment-007-submission.csv
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
    """Resolve only one competition-trained experiment-003 checkpoint."""
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


def instance_auxiliary_targets(
    instances: np.ndarray,
    *,
    boundary_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-instance boundary and normalized interior-distance targets."""
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be a positive odd integer.")
    boundary = np.zeros_like(instances, dtype=np.uint8)
    normalized_distance = np.zeros_like(instances, dtype=np.float32)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_kernel, boundary_kernel),
    )
    instance_ids = np.unique(instances)
    for instance_id in instance_ids[instance_ids != 0]:
        component = (instances == instance_id).astype(np.uint8)
        eroded = cv2.erode(component, kernel, iterations=1)
        boundary |= component & (eroded == 0)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        maximum = float(distance.max())
        if maximum > 0.0:
            normalized_distance = np.maximum(
                normalized_distance,
                distance / maximum,
            )
    return boundary, normalized_distance.astype(np.float16)


def prepare_observations(
    observations: list[dict],
    *,
    image_directory: Path,
    original_size: int,
    model_size: int,
    boundary_kernel: int,
    disk_erosion_pixels: int,
) -> list[dict]:
    """Cache grayscale images and all semantic, instance, and auxiliary targets."""
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
        boundary_masks = []
        distance_targets = []
        for annotation_set in observation["annotation_sets"]:
            semantic, instances = rasterize_instances(
                annotation_set["polygons"],
                original_size=original_size,
                model_size=model_size,
            )
            boundary, distance = instance_auxiliary_targets(
                instances,
                boundary_kernel=boundary_kernel,
            )
            semantic_masks.append(semantic)
            instance_masks.append(instances)
            boundary_masks.append(boundary)
            distance_targets.append(distance)
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
                "boundary_masks": boundary_masks,
                "distance_targets": distance_targets,
                "consensus": consensus,
                "annotator_count": len(semantic_masks),
            }
        )
        if index % 50 == 0 or index == len(observations):
            print(f"Prepared {index}/{len(observations)} physical observations.")
    return prepared


def build_model(base_channels: int):
    """Reconstruct the source U-Net and add boundary and distance heads."""
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

    class BoundaryAwareUNet(nn.Module):
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
            self.boundary_head = nn.Conv2d(channels, 1, 1)
            self.distance_head = nn.Conv2d(channels, 1, 1)

        def forward(self, inputs):
            encoded1 = self.encoder1(inputs)
            encoded2 = self.encoder2(self.pool(encoded1))
            encoded3 = self.encoder3(self.pool(encoded2))
            bottleneck = self.bottleneck(self.pool(encoded3))
            decoded3 = self.decoder3(torch.cat((self.up3(bottleneck), encoded3), dim=1))
            decoded2 = self.decoder2(torch.cat((self.up2(decoded3), encoded2), dim=1))
            decoded1 = self.decoder1(torch.cat((self.up1(decoded2), encoded1), dim=1))
            return {
                "foreground": self.head(decoded1),
                "boundary": self.boundary_head(decoded1),
                "distance": self.distance_head(decoded1),
            }

    return BoundaryAwareUNet()


def make_datasets():
    """Create multi-task training and semantic-validation datasets lazily."""
    import torch

    class BoundaryDataset(torch.utils.data.Dataset):
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
            boundary = sample["boundary_masks"][annotator_index].copy()
            distance = sample["distance_targets"][annotator_index].copy()
            arrays = [image, consensus, annotator, boundary, distance]
            if random.random() < self.augmentation_config["horizontal_flip_probability"]:
                arrays = [np.fliplr(array) for array in arrays]
            if random.random() < self.augmentation_config["vertical_flip_probability"]:
                arrays = [np.flipud(array) for array in arrays]
            if random.random() < self.augmentation_config["rotate_90_probability"]:
                turns = random.randint(1, 3)
                arrays = [np.rot90(array, turns) for array in arrays]
            image, consensus, annotator, boundary, distance = arrays
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
                torch.from_numpy(np.ascontiguousarray(boundary)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(distance)).float().unsqueeze(0),
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

    return BoundaryDataset, ValidationDataset


def base_loss(
    logits,
    targets,
    *,
    positive_weight: float,
    bce_weight: float,
    dice_weight: float,
):
    """Compute weighted BCE plus soft Dice for one binary head."""
    import torch
    from torch.nn import functional

    weight = torch.tensor(positive_weight, device=logits.device)
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight,
    )
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce_weight * bce + dice_weight * dice


def multitask_loss(
    outputs,
    consensus,
    annotator,
    boundary,
    distance,
    *,
    config: dict,
):
    """Combine foreground, boundary, and normalized-distance objectives."""
    import torch
    from torch.nn import functional

    training_config = config["training"]
    foreground_consensus = base_loss(
        outputs["foreground"],
        consensus,
        positive_weight=training_config["foreground_bce_positive_weight"],
        bce_weight=training_config["bce_loss_weight"],
        dice_weight=training_config["dice_loss_weight"],
    )
    foreground_annotator = base_loss(
        outputs["foreground"],
        annotator,
        positive_weight=training_config["foreground_bce_positive_weight"],
        bce_weight=training_config["bce_loss_weight"],
        dice_weight=training_config["dice_loss_weight"],
    )
    target_config = config["targets"]
    foreground = (
        target_config["consensus_weight"] * foreground_consensus
        + target_config["annotator_weight"] * foreground_annotator
    )
    boundary_loss = base_loss(
        outputs["boundary"],
        boundary,
        positive_weight=training_config["boundary_bce_positive_weight"],
        bce_weight=training_config["bce_loss_weight"],
        dice_weight=training_config["dice_loss_weight"],
    )
    distance_probability = torch.sigmoid(outputs["distance"])
    per_pixel_distance = functional.smooth_l1_loss(
        distance_probability,
        distance,
        reduction="none",
    )
    foreground_pixels = (annotator > 0).float()
    background_pixels = 1.0 - foreground_pixels
    foreground_distance = (
        per_pixel_distance * foreground_pixels
    ).sum() / foreground_pixels.sum().clamp_min(1.0)
    background_distance = (
        per_pixel_distance * background_pixels
    ).sum() / background_pixels.sum().clamp_min(1.0)
    foreground_weight = training_config["distance_foreground_weight"]
    distance_loss = (foreground_weight * foreground_distance + background_distance) / (
        foreground_weight + 1.0
    )
    weights = training_config["loss_weights"]
    total = (
        weights["foreground"] * foreground
        + weights["boundary"] * boundary_loss
        + weights["normalized_instance_distance"] * distance_loss
    )
    return total, {
        "foreground": foreground,
        "boundary": boundary_loss,
        "distance": distance_loss,
    }


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
    device,
):
    """Fine-tune one boundary-aware fold and retain its best semantic checkpoint."""
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
        component_losses = {"foreground": [], "boundary": [], "distance": []}
        for images, consensus, annotator, boundary, distance in train_loader:
            images = images.to(device, non_blocking=True)
            consensus = consensus.to(device, non_blocking=True)
            annotator = annotator.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            distance = distance.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss, components = multitask_loss(
                    outputs,
                    consensus,
                    annotator,
                    boundary,
                    distance,
                    config=config,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_losses[name].append(float(value.detach().cpu()))

        model.eval()
        scores = []
        with torch.no_grad():
            for images, targets in validation_loader:
                probabilities = (
                    torch.sigmoid(model(images.to(device, non_blocking=True))["foreground"])
                    .cpu()
                    .numpy()
                )
                scores.extend(semantic_dice(probabilities, targets.numpy()).tolist())
        mean_loss = float(np.mean(losses))
        mean_score = float(np.mean(scores))
        history.append(
            {
                "epoch": epoch,
                "training_loss": mean_loss,
                "training_foreground_loss": float(np.mean(component_losses["foreground"])),
                "training_boundary_loss": float(np.mean(component_losses["boundary"])),
                "training_distance_loss": float(np.mean(component_losses["distance"])),
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
        raise RuntimeError("Boundary-aware fine-tuning did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_state, best_score, best_epoch, history


def predict_with_tta(model, inputs, *, augmentations: list[str]):
    """Average aligned probabilities for all three heads over fixed flips."""
    import torch

    outputs_by_head = {
        "foreground": [],
        "boundary": [],
        "distance": [],
    }
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
        logits = model(transformed)
        for name in outputs_by_head:
            probability = torch.sigmoid(logits[name])
            outputs_by_head[name].append(
                torch.flip(probability, inverse) if inverse else probability
            )
    return {
        name: torch.stack(probabilities).mean(dim=0)
        for name, probabilities in outputs_by_head.items()
    }


def predict_observations(
    model,
    observations: list[dict],
    *,
    config: dict,
    device,
) -> list[dict[str, np.ndarray]]:
    """Predict three probability maps per physical held-out observation."""
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
            probabilities = predict_with_tta(
                model,
                tensor,
                augmentations=config["inference"]["test_time_augmentations"],
            )
            predictions.append(
                {
                    name: probability[0, 0].cpu().numpy()
                    for name, probability in probabilities.items()
                }
            )
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


def foreground_mask(
    probability: np.ndarray,
    image: np.ndarray,
    *,
    config: dict,
    disk_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the common foreground support used by both instance methods."""
    foreground = probability >= config["foreground_probability_threshold"]
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
    return cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config["closing_iterations"],
    )


def filter_labeled_instances(
    labels: np.ndarray,
    foreground_probability: np.ndarray,
    *,
    minimum_area: int,
    minimum_mean_probability: float,
    maximum_area: int,
) -> list[np.ndarray]:
    """Filter and deterministically order labeled instance partitions."""
    candidates = []
    for label in np.unique(labels):
        if label == 0:
            continue
        component = labels == label
        area = int(np.count_nonzero(component))
        if not minimum_area <= area <= maximum_area:
            continue
        mean_probability = float(foreground_probability[component].mean())
        if mean_probability < minimum_mean_probability:
            continue
        y_coordinates, x_coordinates = np.nonzero(component)
        candidates.append(
            (
                -area,
                float(y_coordinates.mean()),
                float(x_coordinates.mean()),
                int(label),
            )
        )
    candidates.sort()
    return [(labels == label).astype(np.uint8) for _, _, _, label in candidates]


def connected_component_instances(
    foreground_probability: np.ndarray,
    image: np.ndarray,
    *,
    config: dict,
    candidate: dict,
    disk_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Separate instances with the semantic connected-components baseline."""
    foreground = foreground_mask(
        foreground_probability,
        image,
        config=config,
        disk_mask=disk_mask,
    )
    _, labels = cv2.connectedComponents(
        foreground,
        connectivity=config["connectivity"],
    )
    return filter_labeled_instances(
        labels,
        foreground_probability,
        minimum_area=candidate["minimum_instance_area"],
        minimum_mean_probability=candidate["minimum_instance_mean_probability"],
        maximum_area=config["max_component_area_at_model_resolution"],
    )


def seeded_partition_instances(
    probabilities: dict[str, np.ndarray],
    image: np.ndarray,
    *,
    config: dict,
    candidate: dict,
    disk_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Partition each foreground component by learned interior seed regions."""
    foreground = foreground_mask(
        probabilities["foreground"],
        image,
        config=config,
        disk_mask=disk_mask,
    )
    seeds = (
        (foreground != 0)
        & (probabilities["boundary"] < candidate["boundary_probability_threshold"])
        & (probabilities["distance"] >= candidate["distance_seed_threshold"])
    )
    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (candidate["seed_closing_kernel"], candidate["seed_closing_kernel"]),
    )
    seeds = cv2.morphologyEx(
        seeds.astype(np.uint8),
        cv2.MORPH_CLOSE,
        seed_kernel,
    )
    seed_count, seed_labels, seed_statistics, _ = cv2.connectedComponentsWithStats(
        seeds,
        connectivity=config["connectivity"],
    )
    cleaned_seeds = np.zeros_like(seeds)
    for seed_label in range(1, seed_count):
        if int(seed_statistics[seed_label, cv2.CC_STAT_AREA]) >= candidate["minimum_seed_area"]:
            cleaned_seeds[seed_labels == seed_label] = 1

    component_count, component_labels, component_statistics, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=config["connectivity"],
    )
    partitions = np.zeros_like(component_labels, dtype=np.int32)
    next_label = 1
    for component_label in range(1, component_count):
        left = int(component_statistics[component_label, cv2.CC_STAT_LEFT])
        top = int(component_statistics[component_label, cv2.CC_STAT_TOP])
        width = int(component_statistics[component_label, cv2.CC_STAT_WIDTH])
        height = int(component_statistics[component_label, cv2.CC_STAT_HEIGHT])
        y_slice = slice(top, top + height)
        x_slice = slice(left, left + width)
        component = component_labels[y_slice, x_slice] == component_label
        component_seeds = cleaned_seeds[y_slice, x_slice] & component.astype(np.uint8)
        local_seed_count, _ = cv2.connectedComponents(
            component_seeds,
            connectivity=config["connectivity"],
        )
        destination = partitions[y_slice, x_slice]
        if local_seed_count <= 2:
            destination[component] = next_label
            next_label += 1
            continue

        distance_input = np.ones(component.shape, dtype=np.uint8)
        distance_input[component_seeds != 0] = 0
        _, nearest_seed = cv2.distanceTransformWithLabels(
            distance_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        for seed_label in np.unique(nearest_seed[component]):
            if seed_label == 0:
                continue
            partition = component & (nearest_seed == seed_label)
            if np.any(partition):
                destination[partition] = next_label
                next_label += 1

    return filter_labeled_instances(
        partitions,
        probabilities["foreground"],
        minimum_area=candidate["minimum_instance_area"],
        minimum_mean_probability=candidate["minimum_instance_mean_probability"],
        maximum_area=config["max_component_area_at_model_resolution"],
    )


def separate_instances(
    probabilities: dict[str, np.ndarray],
    image: np.ndarray,
    *,
    config: dict,
    candidate: dict,
    disk_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Dispatch one frozen OOF candidate to its declared instance method."""
    if candidate["method"] == "connected-components":
        return connected_component_instances(
            probabilities["foreground"],
            image,
            config=config,
            candidate=candidate,
            disk_mask=disk_mask,
        )
    if candidate["method"] == "seeded-partition":
        return seeded_partition_instances(
            probabilities,
            image,
            config=config,
            candidate=candidate,
            disk_mask=disk_mask,
        )
    raise ValueError(f"Unknown instance method: {candidate['method']}")


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
    probabilities: list[dict[str, np.ndarray]],
    *,
    config: dict,
    candidate: dict,
) -> dict:
    """Evaluate one instance method across all grouped OOF annotation sets."""
    aggregate = empty_aggregate()
    by_annotator_count: dict[int, dict] = {}
    for observation, probability_maps in zip(
        observations,
        probabilities,
        strict=True,
    ):
        predictions = separate_instances(
            probability_maps,
            observation["image"],
            config=config["postprocessing"],
            candidate=candidate,
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
    }


def mean_oof_semantic_dice(
    observations: list[dict],
    probabilities: list[dict[str, np.ndarray]],
) -> float:
    """Compute the candidate-independent foreground Dice once."""
    scores = []
    for observation, probability_maps in zip(
        observations,
        probabilities,
        strict=True,
    ):
        for semantic in observation["semantic_masks"]:
            scores.append(
                float(
                    semantic_dice(
                        probability_maps["foreground"][np.newaxis, np.newaxis],
                        semantic[np.newaxis, np.newaxis],
                    )[0]
                )
            )
    return float(np.mean(scores))


def postprocessing_candidates(config: dict) -> list[dict]:
    """Build the frozen connected baseline and seeded-partition grid."""
    postprocessing = config["postprocessing"]
    baseline = postprocessing["connected_components_baseline"]
    candidates = [
        {
            "method": "connected-components",
            "minimum_instance_area": baseline["min_component_area_at_model_resolution"],
            "minimum_instance_mean_probability": baseline["minimum_component_mean_probability"],
        }
    ]
    grid = postprocessing["seeded_partition_grid"]
    for boundary_threshold in grid["boundary_probability_thresholds"]:
        for distance_threshold in grid["distance_seed_thresholds"]:
            for seed_closing_kernel in grid["seed_closing_kernels"]:
                for minimum_seed_area in grid["minimum_seed_areas"]:
                    for minimum_instance_area in grid["minimum_instance_areas"]:
                        for minimum_probability in grid["minimum_instance_mean_probabilities"]:
                            candidates.append(
                                {
                                    "method": "seeded-partition",
                                    "boundary_probability_threshold": float(boundary_threshold),
                                    "distance_seed_threshold": float(distance_threshold),
                                    "seed_closing_kernel": int(seed_closing_kernel),
                                    "minimum_seed_area": int(minimum_seed_area),
                                    "minimum_instance_area": int(minimum_instance_area),
                                    "minimum_instance_mean_probability": float(minimum_probability),
                                }
                            )
    return candidates


def train_fold(
    prepared: list[dict],
    assignments: dict[str, int],
    fold_index: int,
    *,
    config: dict,
    device,
):
    """Fine-tune and predict one immutable grouped fold."""
    import torch

    fold_seed = config["experiment"]["seed"] + fold_index
    seed_everything(fold_seed)
    training = [
        observation
        for observation in prepared
        if assignments[observation["observation_key"]] != fold_index
    ]
    validation = [
        observation
        for observation in prepared
        if assignments[observation["observation_key"]] == fold_index
    ]
    dataset_class, validation_dataset_class = make_datasets()
    training_dataset = dataset_class(
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
    checkpoint = resolve_checkpoint(config, fold_index)
    source_state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = {
        "boundary_head.weight",
        "boundary_head.bias",
        "distance_head.weight",
        "distance_head.bias",
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing source parameters: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source parameters: {incompatible.unexpected_keys}")
    del source_state, incompatible
    print(
        f"Fold {fold_index}: train observations={len(training)}, "
        f"validation observations={len(validation)}, source={checkpoint.name}"
    )
    model, state, best_score, best_epoch, history = train_model(
        model,
        training_dataset,
        train_loader,
        validation_loader,
        config=config,
        device=device,
    )
    probabilities = predict_observations(
        model,
        validation,
        config=config,
        device=device,
    )
    record = {
        "fold": fold_index,
        "seed": fold_seed,
        "training_observations": len(training),
        "validation_observations": len(validation),
        "validation_annotation_sets": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_semantic_dice": best_score,
        "source_checkpoint": checkpoint.name,
        "training_history": history,
    }
    del model
    torch.cuda.empty_cache()
    return state, record, validation, probabilities


def select_candidate(results: list[dict], *, config: dict) -> dict:
    """Select the highest-Dice candidate, preferring the declared feasible set."""
    selection = config["selection"]
    for result in results:
        summary = result["evaluation"]["instance"]
        result["feasible"] = (
            summary["missed"] <= selection["maximum_missed_instances"]
            and summary["extra"] <= selection["maximum_extra_instances"]
        )
    feasible = [result for result in results if result["feasible"]]
    pool = feasible if selection["prefer_feasible_candidates"] and feasible else results
    return max(
        enumerate(pool),
        key=lambda indexed: (
            indexed[1]["evaluation"]["instance"]["mean_penalized_instance_dice"],
            indexed[1]["evaluation"]["instance"]["mean_matched_instance_dice"],
            -indexed[1]["evaluation"]["instance"]["extra"],
            -indexed[1]["evaluation"]["instance"]["missed"],
            -indexed[0],
        ),
    )[1]


def promotion_result(selected: dict, *, config: dict) -> dict:
    """Evaluate the predeclared boundary-aware promotion gate."""
    promotion = config["promotion"]
    summary = selected["evaluation"]["instance"]
    checks = {
        "selected_method": selected["candidate"]["method"] == promotion["required_method"],
        "mean_penalized_instance_dice": summary["mean_penalized_instance_dice"]
        >= promotion["minimum_mean_penalized_instance_dice"],
        "missed_instances": summary["missed"] <= promotion["maximum_missed_instances"],
        "extra_instances": summary["extra"] <= promotion["maximum_extra_instances"],
        "one_to_many": summary["one_to_many"] <= promotion["maximum_one_to_many"],
        "many_to_one": summary["many_to_one"] <= promotion["maximum_many_to_one"],
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
    test_images: dict[str, np.ndarray],
    *,
    config: dict,
    device,
) -> dict[str, dict[str, np.ndarray]]:
    """Average all five boundary-aware models for each probability head."""
    import torch

    heads = ("foreground", "boundary", "distance")
    sums = {
        image_id: {head: np.zeros_like(image, dtype=np.float32) for head in heads}
        for image_id, image in test_images.items()
    }
    folds = config["validation"]["fold_indices"]
    for fold_index in folds:
        model = build_model(config["model"]["base_channels"]).to(device)
        model.load_state_dict(states[fold_index])
        model.eval()
        with torch.no_grad():
            for index, (image_id, image) in enumerate(test_images.items(), start=1):
                tensor = (
                    torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                    / 255.0
                )
                probabilities = predict_with_tta(
                    model,
                    tensor,
                    augmentations=config["inference"]["test_time_augmentations"],
                )
                for head in heads:
                    sums[image_id][head] += probabilities[head][0, 0].cpu().numpy()
                if index % 30 == 0 or index == len(test_images):
                    print(f"Fold {fold_index} test inference {index}/{len(test_images)}.")
        del model
        torch.cuda.empty_cache()
    return {
        image_id: {
            head: probability_sum / len(folds) for head, probability_sum in head_sums.items()
        }
        for image_id, head_sums in sums.items()
    }


def submission_rows(
    probabilities: dict[str, dict[str, np.ndarray]],
    test_images: dict[str, np.ndarray],
    *,
    config: dict,
    candidate: dict,
):
    """Apply the frozen selected instance method and encode final masks."""
    rows = []
    counts_by_image = {}
    for index, (image_id, image) in enumerate(test_images.items(), start=1):
        instances = separate_instances(
            probabilities[image_id],
            image,
            config=config["postprocessing"],
            candidate=candidate,
        )
        seen = set()
        for instance_index, instance in enumerate(instances, start=1):
            resized = cv2.resize(
                instance,
                (
                    config["submission"]["mask_width"],
                    config["submission"]["mask_height"],
                ),
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
    """Train five boundary-aware folds, select on OOF, and infer test data."""
    import torch

    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    if config["experiment"]["mode"] != "boundary-aware-multitask-finetuning":
        raise ValueError("Experiment 007 must remain boundary-aware fine-tuning.")
    if config["model"]["competition_trained_checkpoint_source"] != "experiment-003":
        raise ValueError("Only experiment-003 competition checkpoints are allowed.")
    if config["model"]["pretrained_weights"] or config["model"]["external_labeled_data"]:
        raise ValueError("External supervision is disabled for experiment 007.")
    if not np.isclose(
        config["targets"]["consensus_weight"] + config["targets"]["annotator_weight"],
        1.0,
    ):
        raise ValueError("Foreground target weights must sum to one.")
    if config["selection"]["metric"] != "mean-penalized-instance-dice":
        raise ValueError("OOF selection must use penalized instance Dice.")

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
        boundary_kernel=config["targets"]["boundary_kernel"],
        disk_erosion_pixels=config["postprocessing"]["disk_erosion_pixels"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_hash = hashlib.sha256(EMBEDDED_CONFIG_YAML.encode()).hexdigest()
    print(f"Official data root: {data_root}")
    print(f"Config SHA-256: {config_hash}")
    print(f"Fold fingerprint: {fingerprint}")
    print(f"Training device: {device}")

    states = {}
    fold_records = []
    oof_by_key = {}
    for fold_index in config["validation"]["fold_indices"]:
        state, record, validation, probabilities = train_fold(
            prepared,
            assignments,
            fold_index,
            config=config,
            device=device,
        )
        states[fold_index] = state
        fold_records.append(record)
        for observation, probability_maps in zip(
            validation,
            probabilities,
            strict=True,
        ):
            key = observation["observation_key"]
            if key in oof_by_key:
                raise RuntimeError(f"Duplicate OOF probability maps: {key}")
            oof_by_key[key] = {
                name: probability.astype(np.float16)
                for name, probability in probability_maps.items()
            }
        del probabilities
    expected_keys = {observation["observation_key"] for observation in prepared}
    if set(oof_by_key) != expected_keys:
        raise RuntimeError("Boundary-aware OOF predictions are incomplete.")
    oof_probabilities = [oof_by_key[observation["observation_key"]] for observation in prepared]
    del oof_by_key
    mean_semantic_dice = mean_oof_semantic_dice(
        prepared,
        oof_probabilities,
    )

    candidates = postprocessing_candidates(config)
    if len(candidates) != 97:
        raise RuntimeError(f"Expected 97 frozen candidates, got {len(candidates)}.")
    candidate_results = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = evaluate_oof(
            prepared,
            oof_probabilities,
            config=config,
            candidate=candidate,
        )
        candidate_results.append(
            {
                "candidate": candidate,
                "evaluation": evaluation,
            }
        )
        summary = evaluation["instance"]
        print(
            f"Candidate {index}/{len(candidates)}: "
            f"method={candidate['method']}, "
            f"dice={summary['mean_penalized_instance_dice']:.6f}, "
            f"missed={summary['missed']}, extra={summary['extra']}, "
            f"one_to_many={summary['one_to_many']}, "
            f"many_to_one={summary['many_to_one']}"
        )
    selected = select_candidate(candidate_results, config=config)
    promotion = promotion_result(selected, config=config)
    selected_summary = selected["evaluation"]["instance"]
    oof_metadata = {
        "experiment_id": config["experiment"]["id"],
        "config_sha256": config_hash,
        "fold_fingerprint": fingerprint,
        "checkpoint_source": config["model"]["competition_trained_checkpoint_source"],
        "heads": config["model"]["heads"],
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "selected_candidate": selected["candidate"],
        "selected_candidate_feasible": selected["feasible"],
        "full_oof": selected_summary,
        "full_oof_by_annotator_count": selected["evaluation"]["by_annotator_count"],
        "mean_semantic_dice": mean_semantic_dice,
        "folds": fold_records,
        "promotion_gate": promotion,
        "external_labeled_data": config["model"]["external_labeled_data"],
        "pretrained_weights": config["model"]["pretrained_weights"],
        "test_statistics_used_for_selection": False,
        "physical_observations": len(prepared),
        "annotation_sets": annotation_sets,
        "annotations": annotations,
        "oof_runtime_seconds": time.time() - started,
    }
    oof_path = Path("/kaggle/working/experiment-007-oof-metadata.json")
    oof_path.write_text(
        json.dumps(oof_metadata, indent=2),
        encoding="utf-8",
    )
    print(
        "Selected full OOF: "
        f"method={selected['candidate']['method']}, "
        f"dice={selected_summary['mean_penalized_instance_dice']:.6f}, "
        f"missed={selected_summary['missed']}, "
        f"extra={selected_summary['extra']}, "
        f"one_to_many={selected_summary['one_to_many']}, "
        f"many_to_one={selected_summary['many_to_one']}, "
        f"promotion_gate_passed={promotion['passed']}"
    )
    print(f"OOF evidence written before test inference: {oof_path}")
    del oof_probabilities, candidate_results
    del prepared, observations, assignments

    if sorted(states) != config["validation"]["fold_indices"]:
        raise RuntimeError("Boundary-aware states do not cover all grouped folds.")
    for fold_index, state in states.items():
        torch.save(
            state,
            Path(f"/kaggle/working/experiment-007-fold-{fold_index}.pt"),
        )
    test_images = prepare_test_images(
        data_root / config["data"]["test_images"],
        model_size=config["model"]["input_size"],
    )
    test_probabilities = ensemble_test_probabilities(
        states,
        test_images,
        config=config,
        device=device,
    )
    rows, counts_by_image = submission_rows(
        test_probabilities,
        test_images,
        config=config,
        candidate=selected["candidate"],
    )
    submission_path = Path(config["submission"]["output_path"])
    validate_and_write(
        rows,
        known_ids=set(test_images),
        path=submission_path,
    )
    final_metadata = {
        **oof_metadata,
        "test_images": len(test_images),
        "submission_rows": len(rows),
        "predicted_instances_per_image": counts_by_image,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-007-run-metadata.json")
    metadata_path.write_text(
        json.dumps(final_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Candidate written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This private kernel creates artifacts only; it does not submit.")


if __name__ == "__main__":
    main()
