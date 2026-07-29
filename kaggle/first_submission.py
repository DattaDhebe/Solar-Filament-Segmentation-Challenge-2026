"""Kaggle GPU script for experiment-001, the first submission baseline.

This file is deliberately self-contained because the Kaggle kernels API uploads
only the configured code file. Keep EMBEDDED_CONFIG_YAML synchronized with
configs/experiment-001-first-submission.yaml; a synthetic test enforces that.
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
  id: experiment-001
  name: first-submission-unet
  seed: 20260729
  output_root: outputs/experiments

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

validation:
  strategy: grouped-k-fold
  group_key: file_name
  n_splits: 5
  fold_index: 0
  shuffle: true
  seed: 20260729

model:
  task: instance-segmentation
  architecture: small-unet
  semantic_training_with_instance_separation: true
  input_size: 512
  base_channels: 16
  pretrained_weights: false
  external_labeled_data: false

training:
  epochs: 8
  batch_size: 8
  learning_rate: 0.001
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

postprocessing:
  probability_threshold: 0.5
  closing_kernel: 3
  closing_iterations: 1
  disk_erosion_pixels: 4
  min_component_area_at_model_resolution: 4
  max_component_area_at_model_resolution: 10000
  connectivity: 8

metric:
  primary: mean-matched-instance-dice
  baseline_validation_diagnostic: semantic-dice-at-model-resolution
  diagnostics:
    - dice-distribution
    - iou-distribution
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
  output_path: /kaggle/working/submission.csv
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


def load_examples(annotation_path: Path) -> tuple[list[dict], int]:
    """Load train-only COCO records and associate polygons with image records."""
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    annotations_by_image: dict[object, list[list[float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("Experiment-001 supports the supplied polygon annotations only.")
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


def rasterize_polygons(polygons: list[list[float]], *, height: int, width: int) -> np.ndarray:
    """Rasterize official COCO polygons with pycocotools."""
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_utils.frPyObjects(polygons, height, width)
    return (mask_utils.decode(mask_utils.merge(rles)) != 0).astype(np.uint8)


def prepare_examples(
    examples: list[dict],
    *,
    image_directory: Path,
    original_size: int,
    model_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cache grayscale images and annotation-set masks at model resolution."""
    image_cache: dict[str, np.ndarray] = {}
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
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

        full_mask = rasterize_polygons(
            example["polygons"],
            height=original_size,
            width=original_size,
        )
        mask_image = Image.fromarray(full_mask * 255, mode="L")
        resized_mask = mask_image.resize((model_size, model_size), resampling.NEAREST)
        prepared.append((image_cache[file_name], np.asarray(resized_mask) != 0))

        if index % 100 == 0 or index == len(examples):
            print(f"Prepared {index}/{len(examples)} annotation sets.")
    return prepared


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
            image, mask = self.samples[index]
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
    amp_enabled = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    threshold = config["postprocessing"]["probability_threshold"]
    best_score = -1.0
    best_state = None
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
            }
        )
        print(
            f"Epoch {epoch:02d}/{config['training']['epochs']}: "
            f"loss={mean_loss:.5f}, validation_semantic_dice={mean_score:.5f}"
        )
        if mean_score > best_score:
            best_score = mean_score
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_score, history


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


def infer_submission(model, *, test_directory: Path, config: dict, device):
    """Run final inference once over the supplied test observations."""
    import torch

    model_size = config["model"]["input_size"]
    output_height = config["submission"]["mask_height"]
    output_width = config["submission"]["mask_width"]
    rows = []
    per_image_counts = {}
    test_paths = sorted(test_directory.glob("*.jpeg"))
    if not test_paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_directory}.")

    model.eval()
    with torch.no_grad():
        for index, image_path in enumerate(test_paths, start=1):
            with Image.open(image_path) as image:
                grayscale = image.convert("L")
                resized = grayscale.resize(
                    (model_size, model_size),
                    Image.Resampling.BILINEAR,
                )
                image_array = np.asarray(resized, dtype=np.uint8)

            tensor = (
                torch.from_numpy(image_array.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                / 255.0
            )
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            instances = separate_instances(
                probability,
                image_array,
                config=config["postprocessing"],
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
                    raise ValueError(f"Duplicate predicted instance in {image_path.stem}.")
                seen_counts.add(counts)
                image_rows.append((f"{image_path.stem}_{len(image_rows) + 1}", counts))
            rows.extend(image_rows)
            per_image_counts[image_path.stem] = len(image_rows)
            print(
                f"Inference {index}/{len(test_paths)}: "
                f"{image_path.name}, instances={len(image_rows)}"
            )

    if not rows:
        raise ValueError("The model predicted no filament instances in the complete test set.")
    return rows, per_image_counts, {path.stem for path in test_paths}


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
    """Train, validate, infer once, and create the first-submission artifact."""
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
    fold_index = config["validation"]["fold_index"]
    training_examples = [
        example
        for example in examples
        if assignments[str(example["observation_key"])] != fold_index
    ]
    validation_examples = [
        example
        for example in examples
        if assignments[str(example["observation_key"])] == fold_index
    ]
    training_groups = {str(example["observation_key"]) for example in training_examples}
    validation_groups = {str(example["observation_key"]) for example in validation_examples}
    if training_groups & validation_groups:
        raise RuntimeError("Physical-observation leakage detected across the grouped split.")

    print(
        f"Train annotation sets={len(training_examples)}, "
        f"validation annotation sets={len(validation_examples)}, "
        f"fold fingerprint={fold_fingerprint}"
    )
    prepared = prepare_examples(
        examples,
        image_directory=train_image_directory,
        original_size=config["competition"]["image_height"],
        model_size=config["model"]["input_size"],
    )
    training_samples = [
        sample
        for sample, example in zip(prepared, examples, strict=True)
        if assignments[str(example["observation_key"])] != fold_index
    ]
    validation_samples = [
        sample
        for sample, example in zip(prepared, examples, strict=True)
        if assignments[str(example["observation_key"])] == fold_index
    ]

    dataset_class = make_dataset_class()
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
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = build_model(config["model"]["base_channels"]).to(device)
    model, best_score, history = train_model(
        model,
        train_loader,
        validation_loader,
        config=config,
        device=device,
    )
    checkpoint_path = Path("/kaggle/working/experiment-001-model.pt")
    torch.save(model.state_dict(), checkpoint_path)

    rows, per_image_counts, known_image_ids = infer_submission(
        model,
        test_directory=test_image_directory,
        config=config,
        device=device,
    )
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
        "training_annotation_sets": len(training_examples),
        "validation_annotation_sets": len(validation_examples),
        "best_validation_semantic_dice": best_score,
        "semantic_training_only": True,
        "instance_separation": "connected-components",
        "test_images": len(known_image_ids),
        "submission_rows": len(rows),
        "predicted_instances_per_image": per_image_counts,
        "training_history": history,
        "runtime_seconds": time.time() - started,
    }
    metadata_path = Path("/kaggle/working/experiment-001-run-metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Submission written to {submission_path} with {len(rows)} rows.")
    print(f"Run metadata written to {metadata_path}.")
    print("This kernel creates an artifact only; it does not submit to the competition.")


if __name__ == "__main__":
    main()
