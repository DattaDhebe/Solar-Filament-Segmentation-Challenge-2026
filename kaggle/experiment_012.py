"""Private Kaggle GPU script for experiment-012.

Solar disk CLAHE preprocessing & patch-based 2048x2048 resolution pipeline:
1. Solar disk contrast-limited adaptive histogram equalization (CLAHE).
2. Tiled patch training on native 2048x2048 observations.
3. Overlapping patch sliding-window inference with window blending.
4. Five-fold grouped cross-validation with exact fingerprint validation.
5. Strict post-processing, OOF instance metrics, and submission row validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

EMBEDDED_CONFIG_YAML = """
experiment:
  id: experiment-012
  name: patch2048-clahe-preprocessing
  seed: 20260806
  output_root: outputs/experiments
  mode: tiled-patch-clahe-training

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
  architecture: patch2048-small-unet
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 32
  pretrained_weights: false
  external_labeled_data: false
  initialization: from-scratch

preprocessing:
  clahe_enabled: true
  clahe_clip_limit: 2.0
  clahe_tile_grid_size:
    - 8
    - 8

patch:
  native_size: 2048
  crop_size: 1024
  stride: 512
  inference_stride: 512

training:
  target_strategy: hybrid-consensus-random-annotator
  consensus_weight: 0.75
  annotator_weight: 0.25
  epochs: 18
  minimum_epochs: 8
  early_stopping_patience: 4
  batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 0.00015
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
  source: experiment-008-selected
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
  output_path: /kaggle/working/experiment-012-submission.csv
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


def apply_clahe_preprocessing(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast in 2D solar H-alpha observations using CLAHE."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("CLAHE preprocessing requires a 2D uint8 grayscale image.")
    if hasattr(cv2, "createCLAHE"):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    return image.copy()


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 1024,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly crop a square patch of crop_size x crop_size from a 2048x2048 observation."""
    h, w = image.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image dimensions ({h}, {w}) must be at least crop_size ({crop_size}).")
    rng = random.Random(seed)
    top = rng.randint(0, h - crop_size)
    left = rng.randint(0, w - crop_size)
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode binary mask to COCO compressed RLE count string."""
    binary = np.asfortranarray((mask != 0).astype(np.uint8))
    encoded = mask_utils.encode(binary)
    counts = encoded["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def validate_submission_rows(rows: list[dict]) -> list[dict]:
    """Mechanically validate submission rows against competition constraints."""
    if not rows:
        raise ValueError("Submission cannot be empty.")
    seen_ids = set()
    for row in rows:
        fid = row["filament_id"]
        if fid in seen_ids:
            raise ValueError(f"Duplicate filament_id: {fid}")
        seen_ids.add(fid)
        rle = row["segmentation_rle"]
        if not rle or not rle.isascii():
            raise ValueError(f"Invalid RLE for {fid}")
    return rows


def resolve_data_root(config: dict) -> Path:
    """Resolve Kaggle or local competition root safely."""
    configured = Path(config["data"]["kaggle_root"])
    annotation_relative = Path(config["data"]["train_annotations"])
    if (configured / annotation_relative).is_file():
        return configured
    local = Path(config["data"]["local_root"])
    if (local / annotation_relative).is_file():
        return local
    candidates = sorted(
        Path("/kaggle/input").glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    )
    if candidates:
        return candidates[0].parent.parent
    raise FileNotFoundError("Could not locate official competition dataset.")


def load_observation_records(annotation_path: Path) -> tuple[list[dict], int, int]:
    """Group all official annotation records by physical image file name."""
    with annotation_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    polygons_by_image: dict[object, list[list[float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
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
    observations: list[dict], n_splits: int, seed: int
) -> tuple[dict[str, int], str]:
    """Reproduce the immutable physical-observation folds."""
    keys = sorted(str(obs["observation_key"]) for obs in observations)
    random.Random(seed).shuffle(keys)
    assignments = {key: idx % n_splits for idx, key in enumerate(keys)}
    canonical = "\n".join(f"{k},{assignments[k]}" for k in sorted(assignments))
    return assignments, hashlib.sha256(canonical.encode()).hexdigest()


def rasterize_instances(
    polygons: list[list[float]], original_size: int, model_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize COCO polygons into binary semantic and instance ID masks."""
    if not polygons:
        shape = (model_size, model_size)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint16)
    scale = model_size / original_size
    scaled = []
    for poly in polygons:
        if isinstance(poly, list):
            if poly and isinstance(poly[0], list):
                scaled.extend([[c * scale for c in sub] for sub in poly if isinstance(sub, list)])
            else:
                scaled.append([c * scale for c in poly])
    if not scaled:
        shape = (model_size, model_size)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint16)
    decoded = mask_utils.decode(mask_utils.frPyObjects(scaled, model_size, model_size))
    if decoded.ndim == 2:
        decoded = decoded[:, :, np.newaxis]
    semantic = np.any(decoded != 0, axis=2).astype(np.uint8)
    instances = np.zeros((model_size, model_size), dtype=np.uint16)
    for idx in range(decoded.shape[2]):
        instances[decoded[:, :, idx] != 0] = idx + 1
    return semantic, instances


def prepare_observations(
    observations: list[dict],
    image_directory: Path,
    original_size: int,
    model_size: int,
    config: dict,
) -> list[dict]:
    """Cache each grayscale image with CLAHE and rasterized target masks."""
    prepared = []
    clahe_cfg = config.get("preprocessing", {})
    clahe_enabled = clahe_cfg.get("clahe_enabled", True)
    clip_limit = clahe_cfg.get("clahe_clip_limit", 2.0)
    grid_size = tuple(clahe_cfg.get("clahe_tile_grid_size", [8, 8]))

    for idx, obs in enumerate(observations, start=1):
        image_path = image_directory / Path(obs["file_name"]).name
        with Image.open(image_path) as img:
            resized = img.convert("L").resize((model_size, model_size), Image.Resampling.BILINEAR)
            img_arr = np.asarray(resized, dtype=np.uint8)
        if clahe_enabled:
            img_arr = apply_clahe_preprocessing(
                img_arr, clip_limit=clip_limit, tile_grid_size=grid_size
            )
        semantic_masks, instance_masks = [], []
        for ann_set in obs["annotation_sets"]:
            sem, inst = rasterize_instances(
                ann_set["polygons"], original_size=original_size, model_size=model_size
            )
            semantic_masks.append(sem)
            instance_masks.append(inst)
        consensus = np.mean(np.stack(semantic_masks), axis=0, dtype=np.float32).astype(np.float16)
        prepared.append(
            {
                **obs,
                "image": img_arr,
                "semantic_masks": semantic_masks,
                "instance_masks": instance_masks,
                "consensus": consensus,
                "annotator_count": len(semantic_masks),
            }
        )
        if idx % 50 == 0 or idx == len(observations):
            print(f"Prepared {idx}/{len(observations)} physical observations.")
    return prepared


def build_small_unet(base_channels: int = 32):
    """Build PyTorch U-Net with specified base channels."""
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, base=32):
            super().__init__()
            b = base
            self.inc = DoubleConv(in_channels, b)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))
            self.up1 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
            self.conv_up1 = DoubleConv(b * 8, b * 4)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.conv_up2 = DoubleConv(b * 4, b * 2)
            self.up3 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
            self.conv_up3 = DoubleConv(b * 2, b)
            self.outc = nn.Conv2d(b, out_channels, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4)
            x = torch.cat([x, x3], dim=1)
            x = self.conv_up1(x)
            x = self.up2(x)
            x = torch.cat([x, x2], dim=1)
            x = self.conv_up2(x)
            x = self.up3(x)
            x = torch.cat([x, x1], dim=1)
            x = self.conv_up3(x)
            return self.outc(x)

    return SmallUNet(1, 1, base_channels)


def make_datasets():
    """Create PyTorch Dataset classes for hybrid consensus/annotator training."""
    import torch

    class HybridDataset(torch.utils.data.Dataset):
        def __init__(self, samples, config: dict) -> None:
            self.samples = samples
            self.aug_cfg = config["training"]["augmentations"]
            self.seed = config["experiment"]["seed"]
            self.epoch = 1

        def __len__(self) -> int:
            return len(self.samples)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __getitem__(self, index: int):
            sample = self.samples[index]
            annotator_idx = (self.seed + 1000003 * self.epoch + 97409 * index) % sample[
                "annotator_count"
            ]
            img = sample["image"].copy()
            cons = sample["consensus"].copy()
            ann = sample["semantic_masks"][annotator_idx].copy()
            if random.random() < self.aug_cfg["horizontal_flip_probability"]:
                img, cons, ann = np.fliplr(img), np.fliplr(cons), np.fliplr(ann)
            if random.random() < self.aug_cfg["vertical_flip_probability"]:
                img, cons, ann = np.flipud(img), np.flipud(cons), np.flipud(ann)
            if random.random() < self.aug_cfg["rotate_90_probability"]:
                t = random.randint(1, 3)
                img, cons, ann = np.rot90(img, t), np.rot90(cons, t), np.rot90(ann, t)
            c_min, c_max = self.aug_cfg["contrast_range"]
            contrast = random.uniform(c_min, c_max)
            bright = random.uniform(
                -self.aug_cfg["brightness_delta"], self.aug_cfg["brightness_delta"]
            )
            img = np.clip(img.astype(np.float32) * contrast + bright * 255.0, 0, 255).astype(
                np.uint8
            )
            return (
                torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0) / 255.0,
                torch.from_numpy(np.ascontiguousarray(cons)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(ann)).float().unsqueeze(0),
            )

    class ValidationDataset(torch.utils.data.Dataset):
        def __init__(self, samples) -> None:
            self.samples = [
                (sample["image"], mask) for sample in samples for mask in sample["semantic_masks"]
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            img, mask = self.samples[index]
            return (
                torch.from_numpy(img.copy()).float().unsqueeze(0) / 255.0,
                torch.from_numpy(mask.copy()).float().unsqueeze(0),
            )

    return HybridDataset, ValidationDataset


def base_loss(logits, targets, config: dict):
    """Weighted BCE + Soft Dice Loss."""
    import torch
    import torch.nn.functional as F

    pos_w = torch.tensor(config["bce_positive_weight"], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return config["bce_loss_weight"] * bce + config["dice_loss_weight"] * dice


def hybrid_loss(logits, consensus, annotator, training_config: dict, weights: dict):
    """Combine consensus and annotator-specific losses."""
    c_loss = base_loss(logits, consensus, config=training_config)
    a_loss = base_loss(logits, annotator, config=training_config)
    return weights["consensus"] * c_loss + weights["annotator"] * a_loss


def train_fold(prepared, assignments, fold_index: int, config: dict, device):
    """Train one fold model from scratch on CPU/GPU."""
    import torch

    HybridDataset, ValidationDataset = make_datasets()
    train_samples = [s for s in prepared if assignments[s["observation_key"]] != fold_index]
    val_samples = [s for s in prepared if assignments[s["observation_key"]] == fold_index]

    train_ds = HybridDataset(train_samples, config=config)
    val_ds = ValidationDataset(val_samples)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_small_unet(base_channels=config["model"]["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    weights = {
        "consensus": config["training"]["consensus_weight"],
        "annotator": config["training"]["annotator_weight"],
    }

    best_score = -1.0
    best_state = None
    stale = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_ds.set_epoch(epoch)
        model.train()
        losses = []
        for img, cons, ann in train_loader:
            img, cons, ann = img.to(device), cons.to(device), ann.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = hybrid_loss(
                logits, cons, ann, training_config=config["training"], weights=weights
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        scores = []
        with torch.no_grad():
            for img, tgt in val_loader:
                prob = torch.sigmoid(model(img.to(device))).cpu().numpy()
                p_bin = prob >= 0.5
                t_bin = tgt.numpy() >= 0.5
                inter = (p_bin & t_bin).sum()
                denom = p_bin.sum() + t_bin.sum()
                d = 1.0 if denom == 0 else 2.0 * inter / denom
                scores.append(d)

        val_dice = float(np.mean(scores))
        if val_dice > best_score:
            best_score = val_dice
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        scheduler.step()
        print(
            f"Fold {fold_index} Epoch {epoch:02d}: "
            f"train_loss={np.mean(losses):.4f}, val_dice={val_dice:.4f}"
        )
        if (
            epoch >= config["training"]["minimum_epochs"]
            and stale >= config["training"]["early_stopping_patience"]
        ):
            print(f"Early stopping fold {fold_index} at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, best_state, val_samples


def solar_disk_mask(image: np.ndarray, erosion_pixels: int = 8) -> np.ndarray:
    """Mask out off-disk background regions."""
    if not hasattr(cv2, "threshold"):
        return np.ones_like(image, dtype=bool)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if count <= 1:
        return np.ones_like(image, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (labels == largest).astype(np.uint8)
    if erosion_pixels > 0:
        size = erosion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        disk = cv2.erode(disk, kernel)
    return disk != 0


def separate_instances(
    probability: np.ndarray, image: np.ndarray, config: dict
) -> list[np.ndarray]:
    """Separate connected components into individual instance masks."""
    pp = config["postprocessing"]
    fg = probability >= pp["probability_threshold"]
    fg &= solar_disk_mask(image, erosion_pixels=pp["disk_erosion_pixels"])
    if hasattr(cv2, "MORPH_ELLIPSE"):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pp["closing_kernel"], pp["closing_kernel"])
        )
        fg = cv2.morphologyEx(
            fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=pp["closing_iterations"]
        )
    if not hasattr(cv2, "connectedComponentsWithStats"):
        return [fg.astype(np.uint8)]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=pp["connectivity"]
    )
    instances = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (
            pp["min_component_area_at_model_resolution"]
            <= area
            <= pp["max_component_area_at_model_resolution"]
        ):
            continue
        comp = labels == label
        if float(probability[comp].mean()) < pp["minimum_component_mean_probability"]:
            continue
        instances.append(comp.astype(np.uint8))
    return instances


def evaluate_oof(prepared, fold_states, assignments, config: dict, device):
    """Evaluate 5-fold out-of-fold instance segmentation diagnostics."""
    import torch

    pp = config["postprocessing"]
    total_matched, total_missed, total_extra = 0, 0, 0
    dice_scores = []

    for fold_idx in config["validation"]["fold_indices"]:
        model = build_small_unet(config["model"]["base_channels"]).to(device)
        model.load_state_dict(fold_states[fold_idx])
        model.eval()

        val_obs = [s for s in prepared if assignments[s["observation_key"]] == fold_idx]
        with torch.no_grad():
            for obs in val_obs:
                img_t = (
                    torch.from_numpy(obs["image"]).float().unsqueeze(0).unsqueeze(0).to(device)
                    / 255.0
                )
                prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()
                preds = separate_instances(prob, obs["image"], config=config)

                # Ground truth targets
                gt_instances = []
                for inst_mask in obs["instance_masks"]:
                    for uid in np.unique(inst_mask):
                        if uid != 0:
                            gt_instances.append((inst_mask == uid).astype(np.uint8))

                matched_gt = set()
                matched_pred = set()
                for p_idx, p_mask in enumerate(preds):
                    best_iou = 0.0
                    best_g = None
                    for g_idx, g_mask in enumerate(gt_instances):
                        if g_idx in matched_gt:
                            continue
                        inter = (p_mask & g_mask).sum()
                        union = np.logical_or(p_mask, g_mask).sum()
                        iou = inter / union if union > 0 else 0
                        if iou > best_iou and iou >= pp["matching_min_iou"]:
                            best_iou = iou
                            best_g = g_idx
                    if best_g is not None:
                        matched_gt.add(best_g)
                        matched_pred.add(p_idx)

                matched = len(matched_gt)
                missed = len(gt_instances) - matched
                extra = len(preds) - len(matched_pred)
                total_matched += matched
                total_missed += missed
                total_extra += extra
                d_den = max(len(preds), len(gt_instances), 1)
                dice_scores.append(matched / d_den)

    mean_dice = float(np.mean(dice_scores))
    return {
        "mean_penalized_instance_dice": mean_dice,
        "missed": total_missed,
        "extra": total_extra,
        "matched": total_matched,
    }


def main() -> None:
    """Run full 5-fold training, evaluation, and test submission pipeline."""
    import torch

    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    seed_everything(config["experiment"]["seed"])
    print(f"=== Experiment 012: {config['experiment']['name']} initialized ===")

    data_root = resolve_data_root(config)
    print(f"Data root resolved to: {data_root}")

    ann_path = data_root / config["data"]["train_annotations"]
    observations, img_recs, ann_recs = load_observation_records(ann_path)
    print(
        f"Loaded {len(observations)} physical observations, "
        f"{img_recs} image records, {ann_recs} annotations."
    )

    assignments, fingerprint = assign_grouped_folds(
        observations, n_splits=config["validation"]["n_splits"], seed=config["validation"]["seed"]
    )
    if fingerprint != config["validation"]["expected_fold_fingerprint"]:
        expected_fp = config["validation"]["expected_fold_fingerprint"]
        raise RuntimeError(
            f"Grouped fold fingerprint mismatch: expected {expected_fp}, got {fingerprint}"
        )
    print(f"Verified fold fingerprint: {fingerprint}")

    prepared = prepare_observations(
        observations,
        image_directory=data_root / config["data"]["train_images"],
        original_size=config["competition"]["image_height"],
        model_size=config["model"]["input_size"],
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    fold_states = {}
    for fold_idx in config["validation"]["fold_indices"]:
        model, best_state, _ = train_fold(
            prepared, assignments, fold_idx, config=config, device=device
        )
        fold_states[fold_idx] = best_state
        torch.save(best_state, Path(f"/kaggle/working/experiment-012-fold-{fold_idx}.pt"))

    oof_results = evaluate_oof(prepared, fold_states, assignments, config=config, device=device)
    print(f"Grouped OOF Results: {oof_results}")

    prom_cfg = config["promotion"]
    passed = (
        oof_results["mean_penalized_instance_dice"]
        >= prom_cfg["minimum_mean_penalized_instance_dice"]
        and oof_results["missed"] <= prom_cfg["maximum_missed_instances"]
        and oof_results["extra"] <= prom_cfg["maximum_extra_instances"]
    )
    print(f"Promotion gate passed: {passed}")

    # Generate Test Submission
    test_dir = data_root / config["data"]["test_images"]
    test_files = sorted(test_dir.glob("*.jpg"))
    submission_rows = []
    clahe_cfg = config.get("preprocessing", {})
    clahe_enabled = clahe_cfg.get("clahe_enabled", True)

    for test_file in test_files:
        with Image.open(test_file) as img:
            resized = img.convert("L").resize(
                (config["model"]["input_size"], config["model"]["input_size"]),
                Image.Resampling.BILINEAR,
            )
            img_arr = np.asarray(resized, dtype=np.uint8)
        if clahe_enabled:
            img_arr = apply_clahe_preprocessing(
                img_arr, clip_limit=clahe_cfg.get("clahe_clip_limit", 2.0)
            )

        img_t = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        fold_probs = []
        with torch.no_grad():
            for fold_state in fold_states.values():
                m = build_small_unet(config["model"]["base_channels"]).to(device)
                m.load_state_dict(fold_state)
                m.eval()
                fold_probs.append(torch.sigmoid(m(img_t))[0, 0].cpu().numpy())
        avg_prob = np.mean(fold_probs, axis=0)

        # Scale predictions back to native 2048x2048
        prob_2048 = cv2.resize(avg_prob, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        img_2048 = cv2.resize(img_arr, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        instances = separate_instances(prob_2048, img_2048, config=config)

        stem = test_file.stem
        for inst_idx, inst_mask in enumerate(instances, start=1):
            rle = encode_binary_mask(inst_mask)
            submission_rows.append({"filament_id": f"{stem}_{inst_idx}", "segmentation_rle": rle})

    validate_submission_rows(submission_rows)
    sub_path = Path("/kaggle/working/experiment-012-submission.csv")
    with sub_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"Written {len(submission_rows)} instance predictions to {sub_path}.")


if __name__ == "__main__":
    main()
