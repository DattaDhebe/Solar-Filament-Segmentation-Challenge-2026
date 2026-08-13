"""Standalone Kaggle Submission script for experiment-013 (Attention U-Net).
Loads pre-trained 5-fold checkpoints from input dataset and generates submission.csv.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from pycocotools import mask as mask_utils

EMBEDDED_CONFIG_YAML = """
experiment:
  id: experiment-013
  name: deep-supervision-attention-unet-watershed
  seed: 20260806
  output_root: outputs/experiments
  mode: attention-unet-watershed-training

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
  expected_fingerprint: 69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7

model:
  architecture: AttentionUNet
  semantic_training_with_instance_separation: true
  input_size: 1024
  base_channels: 32
  pretrained_weights: false
  external_labeled_data: false
  initialization: from-scratch
  attention_gates: true
  deep_supervision: true
  distance_transform_head: true

training:
  target_strategy: hybrid-consensus-random-annotator
  consensus_weight: 0.75
  annotator_weight: 0.25
  epochs: 15
  minimum_epochs: 8
  early_stopping_patience: 4
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 0.00015
  weight_decay: 0.0001
  tversky_alpha: 0.3
  tversky_beta: 0.7
  tversky_loss_weight: 1.0
  bce_loss_weight: 1.0
  bce_positive_weight: 4.0
  distance_loss_weight: 0.5
  deep_supervision_weight: 0.3
  num_workers: 2
  mixed_precision: true
  augmentations:
    horizontal_flip_probability: 0.5
    vertical_flip_probability: 0.5
    rotate_90_probability: 0.5
    brightness_delta: 0.05

postprocessing:
  probability_threshold: 0.35
  watershed_seed_threshold: 0.25
  distance_smooth_kernel: 5
  min_component_area: 48
  max_component_area: 250000

submission:
  output_path: /kaggle/working/submission.csv
  target_shape:
    - 2048
    - 2048

promotion_gate:
  minimum_mean_penalized_instance_dice: 0.40
  maximum_missed_instances: 1500
  maximum_extra_instances: 5000
"""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_data_root(config: dict) -> Path:
    candidates = [
        Path(config["data"]["kaggle_root"]),
        Path(config["data"]["local_root"]),
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("../data/raw/MAGFiLO_1.0_Kaggle_2026"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Data root not found. Tried: {candidates}")


def build_model(config: dict):
    base_channels = config["model"]["base_channels"]

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)

    class AttentionGate(nn.Module):
        def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
                nn.GroupNorm(8, F_int),
            )
            self.W_l = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
                nn.GroupNorm(8, F_int),
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
                nn.GroupNorm(1, 1),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_l(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class AttentionUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = ConvBlock(1, c)
            self.enc2 = ConvBlock(c, c * 2)
            self.enc3 = ConvBlock(c * 2, c * 4)
            self.enc4 = ConvBlock(c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = ConvBlock(c * 8, c * 16)

            self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
            self.att4 = AttentionGate(c * 8, c * 8, c * 4)
            self.dec4 = ConvBlock(c * 16, c * 8)

            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.att3 = AttentionGate(c * 4, c * 4, c * 2)
            self.dec3 = ConvBlock(c * 8, c * 4)

            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.att2 = AttentionGate(c * 2, c * 2, c)
            self.dec2 = ConvBlock(c * 4, c * 2)

            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.att1 = AttentionGate(c, c, c // 2)
            self.dec1 = ConvBlock(c * 2, c)

            self.mask_head = nn.Conv2d(c, 1, 1)
            self.dist_head = nn.Conv2d(c, 1, 1)
            self.aux_head = nn.Conv2d(c * 2, 1, 1)

        def forward(self, inputs):
            e1 = self.enc1(inputs)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            b = self.bottleneck(self.pool(e4))

            d4 = self.up4(b)
            a4 = self.att4(d4, e4)
            d4 = self.dec4(torch.cat([d4, a4], dim=1))

            d3 = self.up3(d4)
            a3 = self.att3(d3, e3)
            d3 = self.dec3(torch.cat([d3, a3], dim=1))

            d2 = self.up2(d3)
            a2 = self.att2(d2, e2)
            d2 = self.dec2(torch.cat([d2, a2], dim=1))

            d1 = self.up1(d2)
            a1 = self.att1(d1, e1)
            d1 = self.dec1(torch.cat([d1, a1], dim=1))

            mask_logit = self.mask_head(d1)
            dist_pred = torch.sigmoid(self.dist_head(d1))
            aux_logit = self.aux_head(d2)
            return mask_logit, dist_pred, aux_logit

    return AttentionUNet()


def prepare_test_images(test_dir: Path, *, model_size: int = 1024) -> dict[str, np.ndarray]:
    images = {}
    paths = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.jpeg"))
    for path in paths:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Failed to read image at {path}")
        resized = cv2.resize(raw, (model_size, model_size), interpolation=cv2.INTER_AREA)
        images[path.name] = resized
    return images


def predict_observation(model, image: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    def forward_fn(x):
        mask_logit, dist_pred, _ = model(x)
        mask_prob = torch.sigmoid(mask_logit).squeeze(0).squeeze(0).cpu().numpy()
        dist_map = dist_pred.squeeze(0).squeeze(0).cpu().numpy()
        return mask_prob, dist_map

    p0, d0 = forward_fn(tensor)

    # TTA horizontal flip
    tensor_h = torch.flip(tensor, dims=[3])
    ph, dh = forward_fn(tensor_h)
    ph = np.fliplr(ph)
    dh = np.fliplr(dh)

    # TTA vertical flip
    tensor_v = torch.flip(tensor, dims=[2])
    pv, dv = forward_fn(tensor_v)
    pv = np.flipud(pv)
    dv = np.flipud(dv)

    mask_prob = (p0 + ph + pv) / 3.0
    dist_map = (d0 + dh + dv) / 3.0
    return mask_prob, dist_map


def ensemble_test_probabilities(
    states: dict[int, dict],
    images: dict[str, np.ndarray],
    *,
    config: dict,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    models = []
    for state in states.values():
        model = build_model(config).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    test_mask_probs = []
    test_dist_maps = []
    with torch.no_grad():
        for filename in images:
            image = images[filename]
            p_sum = None
            d_sum = None
            for model in models:
                p, d = predict_observation(model, image, device=device)
                if p_sum is None:
                    p_sum = p
                    d_sum = d
                else:
                    p_sum += p
                    d_sum += d
            test_mask_probs.append(p_sum / len(models))
            test_dist_maps.append(d_sum / len(models))
    return test_mask_probs, test_dist_maps


def separate_instances(
    mask_prob: np.ndarray,
    dist_map: np.ndarray,
    *,
    config: dict,
) -> list[np.ndarray]:
    prob_thresh = config["postprocessing"]["probability_threshold"]
    seed_thresh = config["postprocessing"]["watershed_seed_threshold"]
    kernel_size = config["postprocessing"]["distance_smooth_kernel"]
    min_area = config["postprocessing"]["min_component_area"]
    max_area = config["postprocessing"]["max_component_area"]

    binary_mask = (mask_prob >= prob_thresh).astype(np.uint8)
    if binary_mask.sum() == 0:
        return []

    if kernel_size > 1:
        smoothed_dist = cv2.GaussianBlur(dist_map, (kernel_size, kernel_size), 0)
    else:
        smoothed_dist = dist_map

    seeds = (smoothed_dist >= seed_thresh).astype(np.uint8) * binary_mask
    num_labels, markers = cv2.connectedComponents(seeds)

    if num_labels <= 1:
        num_labels, markers = cv2.connectedComponents(binary_mask)
        if num_labels <= 1:
            return []

    # Watershed separation
    img_dummy = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img_dummy, markers.copy())

    instances = []
    target_h, target_w = config["submission"]["target_shape"]
    orig_h, orig_w = mask_prob.shape

    for label_id in range(1, num_labels):
        inst_mask = (markers_ws == label_id).astype(np.uint8)
        area = inst_mask.sum()
        if area < min_area or area > max_area:
            continue

        if (orig_h, orig_w) != (target_h, target_w):
            inst_mask = cv2.resize(
                inst_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

        instances.append(inst_mask)

    return instances


def encode_rle(mask: np.ndarray) -> str:
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")


def submission_rows(
    mask_probs: list[np.ndarray],
    dist_maps: list[np.ndarray],
    test_images: dict[str, np.ndarray],
    *,
    config: dict,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = []
    counts_by_image = {}

    for (filename, image), prob, dist in zip(test_images.items(), mask_probs, dist_maps, strict=True):
        instances = separate_instances(prob, dist, config=config)
        counts_by_image[filename] = len(instances)

        for inst in instances:
            rle_str = encode_rle(inst)
            rows.append({
                "ImageId": filename,
                "EncodedPixels": rle_str,
            })

    return rows, counts_by_image


def validate_and_write(rows: list[dict[str, str]], *, known_ids: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageId", "EncodedPixels"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.time()
    config = yaml.safe_load(EMBEDDED_CONFIG_YAML)
    seed_everything(config["experiment"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Submission Engine Device: {device}")

    # Check candidate directories for pre-trained weight files
    candidate_dirs = [
        Path("/kaggle/input/solar-filament-attention-unet-watershed"),
        Path("/kaggle/input/solar-filament-attention-unet-watershed/kaggle/working"),
        Path("../kaggle_output_check/exp13"),
        Path("kaggle_output_check/exp13"),
        Path("/kaggle/working"),
    ]

    weights_dir = None
    for d in candidate_dirs:
        if d.exists() and (d / "experiment-013-fold-0.pt").exists():
            weights_dir = d
            break

    if weights_dir is None:
        raise FileNotFoundError(f"Could not locate experiment-013 fold weights in candidates: {candidate_dirs}")

    print(f"Loading 5-fold weights from: {weights_dir}")
    states = {}
    for fold in range(5):
        w_path = weights_dir / f"experiment-013-fold-{fold}.pt"
        print(f"  Loading {w_path.name}...")
        states[fold] = torch.load(w_path, map_location=device)

    data_root = resolve_data_root(config)
    model_size = config["model"]["input_size"]

    test_images = prepare_test_images(
        data_root / config["data"]["test_images"],
        model_size=model_size,
    )
    print(f"Prepared {len(test_images)} test images.")

    print("Running 5-fold TTA test inference...")
    test_mask_probs, test_dist_maps = ensemble_test_probabilities(
        states,
        test_images,
        config=config,
        device=device,
    )

    print("Running Watershed post-processing & RLE encoding...")
    rows, counts_by_image = submission_rows(
        test_mask_probs, test_dist_maps, test_images, config=config
    )

    submission_path = Path(config["submission"]["output_path"]) if Path("/kaggle/working").exists() else Path("submission.csv")
    validate_and_write(rows, known_ids=set(test_images), path=submission_path)

    print(f"Candidate written to {submission_path} with {len(rows)} rows.")
    print(f"Predicted instances per image summary: {counts_by_image}")
    print(f"Runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
