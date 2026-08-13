"""
Kaggle Ensemble Inference Script
Combines Attention U-Net (experiment_013) and UNet++ (experiment_014) predictions.
"""

import json
import time
from pathlib import Path

# Import the architectures and helper functions from the individual scripts
import experiment_013 as exp13
import experiment_014 as exp14
import numpy as np
import torch
import yaml

# Define paths to your weights. Update these based on your Kaggle environment!
# Example Kaggle paths: "/kaggle/input/experiment-013-weights/"
EXP13_WEIGHTS_DIR = Path("../kaggle_output_check/exp13")
EXP14_WEIGHTS_DIR = Path("../kaggle_output_check/exp14")


def load_ensemble_states(device: torch.device):
    """Load all 5 folds of both models."""
    states_13 = {}
    states_14 = {}
    
    print("Loading Attention U-Net (013) weights...")
    for fold in range(5):
        path = EXP13_WEIGHTS_DIR / f"experiment-013-fold-{fold}.pt"
        if path.exists():
            states_13[fold] = torch.load(path, map_location=device)
        else:
            print(f"  -> Missing {path}")
            
    print("Loading UNet++ (014) weights...")
    for fold in range(5):
        path = EXP14_WEIGHTS_DIR / f"experiment-014-fold-{fold}.pt"
        if path.exists():
            states_14[fold] = torch.load(path, map_location=device)
        else:
            print(f"  -> Missing {path}")
            
    return states_13, states_14


def ensemble_test_probabilities(
    states_13: dict,
    states_14: dict,
    images: list[np.ndarray],
    config_13: dict,
    config_14: dict,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate 50/50 blended probabilities using TTA from both models."""
    
    # 1. Get raw predictions from 013
    print("\nRunning TTA for Attention U-Net (experiment_013)...")
    probs_13, dists_13 = exp13.ensemble_test_probabilities(
        states_13, images, config=config_13, device=device
    )
    
    # 2. Get raw predictions from 014
    print("\nRunning TTA for UNet++ (experiment_014)...")
    probs_14, dists_14 = exp14.ensemble_test_probabilities(
        states_14, images, config=config_14, device=device
    )
    
    # 3. Blend them 50/50
    print("\nBlending predictions...")
    blended_probs = []
    blended_dists = []
    for p13, p14, d13, d14 in zip(probs_13, probs_14, dists_13, dists_14, strict=True):
        blended_probs.append(0.5 * p13 + 0.5 * p14)
        blended_dists.append(0.5 * d13 + 0.5 * d14)
        
    return blended_probs, blended_dists


def main():
    started = time.time()
    config_13 = yaml.safe_load(exp13.EMBEDDED_CONFIG_YAML)
    config_14 = yaml.safe_load(exp14.EMBEDDED_CONFIG_YAML)
    
    exp14.seed_everything(config_14["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ensemble Inference Device: {device}")
    
    # Resolve data root (assume both use the same competition data)
    data_root = exp14.resolve_data_root(config_14)
    model_size = config_14["model"]["input_size"]
    
    # Load test images
    print("Preparing test images...")
    test_images = exp14.prepare_test_images(
        data_root / config_14["data"]["test_images"],
        model_size=model_size,
    )
    
    # Load Weights
    states_13, states_14 = load_ensemble_states(device)
    if not states_13 or not states_14:
        print("ERROR: Weights not found. Update EXP13_WEIGHTS_DIR and EXP14_WEIGHTS_DIR.")
        return
        
    # Get blended predictions
    test_mask_probs, test_dist_maps = ensemble_test_probabilities(
        states_13,
        states_14,
        test_images,
        config_13,
        config_14,
        device=device,
    )
    
    # Run post-processing and generate submission rows
    # (We use config_14's post-processing thresholds which we previously optimized)
    print("\nRunning Watershed Post-processing and RLE encoding...")
    rows, counts_by_image = exp14.submission_rows(
        test_mask_probs, test_dist_maps, test_images, config=config_14
    )
    
    # Write submission
    submission_path = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
    exp14.validate_and_write(rows, known_ids=set(test_images), path=submission_path)
    
    final_metadata = {
        "ensemble": "AttentionUNet + UNet++",
        "weights_13_folds_loaded": list(states_13.keys()),
        "weights_14_folds_loaded": list(states_14.keys()),
        "test_images": len(test_images),
        "submission_rows": len(rows),
        "predicted_instances_per_image": counts_by_image,
        "runtime_seconds": time.time() - started,
    }
    
    metadata_path = Path("/kaggle/working/ensemble-run-metadata.json") if Path("/kaggle/working").exists() else Path("ensemble-run-metadata.json")
    metadata_path.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    
    print(f"\\nEnsemble Submission generated at {submission_path} with {len(rows)} instances.")
    print(f"Total Runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
