"""
Generate submission CSV for Experiment 013 (Attention U-Net)
using the 5-fold trained model weights downloaded from Kaggle.
"""

import sys
from pathlib import Path

import torch
import yaml

# Add project root and kaggle directory to import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kaggle"))

import experiment_013 as exp13


def main():
    print("=== Generating Experiment 013 Submission CSV ===")
    config = yaml.safe_load(exp13.EMBEDDED_CONFIG_YAML)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load 5 fold weights
    weights_dir = PROJECT_ROOT / "kaggle_output_check" / "exp13"
    states = {}
    for fold in range(5):
        weight_path = weights_dir / f"experiment-013-fold-{fold}.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing fold weight: {weight_path}")
        print(f"Loading fold {fold} weight from {weight_path.name}...")
        states[fold] = torch.load(weight_path, map_location=device)

    # Local data root
    data_root = PROJECT_ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    test_dir = data_root / config["data"]["test_images"]
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model_size = config["model"]["input_size"]
    print(f"Preparing test images from {test_dir} (model_size={model_size})...")
    test_images = exp13.prepare_test_images(test_dir, model_size=model_size)
    print(f"Loaded {len(test_images)} test images.")

    print("Running 5-fold TTA test probability ensembling...")
    test_mask_probs, test_dist_maps = exp13.ensemble_test_probabilities(
        states,
        test_images,
        config=config,
        device=device,
    )

    print("Running Watershed post-processing & RLE encoding...")
    rows, counts_by_image = exp13.submission_rows(
        test_mask_probs, test_dist_maps, test_images, config=config
    )

    output_dir = PROJECT_ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission_exp13.csv"

    print(f"Validating and writing submission to {submission_path}...")
    exp13.validate_and_write(rows, known_ids=set(test_images), path=submission_path)
    print(f"SUCCESS: Generated {len(rows)} instances across {len(test_images)} test images.")
    print(f"Saved to: {submission_path}")


if __name__ == "__main__":
    main()
