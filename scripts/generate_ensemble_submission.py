"""
Ensemble Submission Generator Script
Blends 5-fold Attention U-Net (exp013) and 5-fold UNet++ (exp014) predictions.
"""

import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root and kaggle directory to import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kaggle"))

# pyrefly: ignore [missing-import]
import experiment_013 as exp13

# pyrefly: ignore [missing-import]
import experiment_014 as exp14


def main():
    started = time.time()
    print("=== Solar Filament Ensemble Submission Generator ===")
    
    config_13 = yaml.safe_load(exp13.EMBEDDED_CONFIG_YAML)
    config_14 = yaml.safe_load(exp14.EMBEDDED_CONFIG_YAML)
    
    exp14.seed_everything(config_14["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths to local fold weights
    exp13_dir = PROJECT_ROOT / "kaggle_output_check" / "exp13"
    exp14_dir = PROJECT_ROOT / "kaggle_output_check" / "exp14"

    print("Loading Attention U-Net (exp013) fold weights...")
    states_13 = {}
    for fold in range(5):
        path = exp13_dir / f"experiment-013-fold-{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing exp13 weight: {path}")
        print(f"  Loaded fold {fold} weight from {path.name}")
        states_13[fold] = torch.load(path, map_location=device)

    print("Loading UNet++ (exp014) fold weights...")
    states_14 = {}
    for fold in range(5):
        path = exp14_dir / f"experiment-013-fold-{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing exp14 weight: {path}")
        print(f"  Loaded fold {fold} weight from {path.name}")
        states_14[fold] = torch.load(path, map_location=device)

    # Local data root
    data_root = PROJECT_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    test_dir = data_root / config_14["data"]["test_images"]
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model_size = config_14["model"]["input_size"]
    print(f"\nPreparing test images from {test_dir} (model_size={model_size})...")
    test_images = exp14.prepare_test_images(test_dir, model_size=model_size)
    print(f"Prepared {len(test_images)} test images.")

    print("\n1. Running TTA test inference for Attention U-Net (58.53% Semantic Dice)...")
    probs_13, dists_13 = exp13.ensemble_test_probabilities(
        states_13, test_images, config=config_13, device=device
    )

    print("\n2. Running TTA test inference for UNet++ (61.71% Semantic Dice)...")
    probs_14, dists_14 = exp14.ensemble_test_probabilities(
        states_14, test_images, config=config_14, device=device
    )

    print("\n3. Blending predictions (50/50 ensemble)...")
    blended_probs = {}
    blended_dists = {}
    for image_id in probs_13:
        blended_probs[image_id] = 0.5 * probs_13[image_id] + 0.5 * probs_14[image_id]
        blended_dists[image_id] = 0.5 * dists_13[image_id] + 0.5 * dists_14[image_id]

    print("\n4. Running Watershed post-processing & RLE encoding...")
    rows, counts_by_image = exp14.submission_rows(
        blended_probs, blended_dists, test_images, config=config_14
    )

    output_dir = PROJECT_ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission_ensemble.csv"

    print(f"\nWriting submission to {submission_path}...")
    exp14.validate_and_write(rows, known_ids=set(test_images), path=submission_path)

    print(f"\nSUCCESS: Generated {len(rows)} instances across {len(test_images)} test images.")
    print(f"Saved to: {submission_path}")
    print(f"Total Ensemble Runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
