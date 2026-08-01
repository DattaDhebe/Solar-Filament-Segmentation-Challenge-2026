# Solar Filament Segmentation Challenge 2026

Active competition workspace for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The task is instance segmentation of solar filaments in 2048 × 2048 grayscale
GONG H-alpha observations.

## Challenge overview

The Solar Filament Segmentation Challenge 2026 is a computer-vision competition
focused on identifying individual solar filaments in full-disk H-alpha images.
Solar filaments appear as dark, elongated structures against the solar disk and
can be thin, fragmented, and morphologically complex, making accurate instance
segmentation more difficult than ordinary foreground detection.

Participants must predict a separate binary mask for every filament instance in
each test observation. Predictions are submitted as COCO compressed
run-length-encoded masks at the original 2048 × 2048 image resolution. The
competition evaluates segmentation overlap and instance behavior, including
Dice, IoU, missed or extra detections, fragmentation, and merged instances.

This repository provides a reproducible pipeline for data validation, grouped
cross-validation, model experiments, instance-aware post-processing, diagnostic
evaluation, and submission generation while maintaining strict separation
between training and test observations.

## Current status

- Kaggle access and competition entry are verified through the configured CLI.
- The full official dataset is downloaded to Git-ignored local storage.
- All 707 training and 180 test images are verified as 2048 × 2048 grayscale
  JPEGs, with no exact filename or SHA-256 overlap between the splits.
- The COCO JSON contains 1,154 annotation-set image records for 707 physical
  observations and 8,199 filament annotations.
- The isolated Python environment, repository safeguards, and synthetic tests
  are ready.
- Experiment 001 trained a small U-Net from scratch on one fixed grouped fold
  and reached 0.6403 semantic Dice at 512 × 512. This is a local diagnostic, not
  the organizer's complete instance-matching score.
- Experiment 002 trained five grouped 1024 × 1024 models from scratch and
  reached 0.6481 mean fold semantic Dice, 0.4304 penalized OOF instance Dice,
  and a verified public score of 0.62.
- Experiment 003 fine-tuned all five models using soft annotator-consensus
  targets, then applied TTA and validation-selected instance refinement. It
  reached 0.4901 penalized OOF instance Dice and a verified public score of
  0.66.
- Experiment 004 reduced extras but increased misses; Kaggle cancelled it during
  final encoding, leaving no complete candidate, and its OOF promotion gate
  failed.
- Experiment 005 completed with 0.4905 penalized OOF instance Dice, 1,609
  misses, and 2,309 extras. Its recall improved over experiment 003, but the
  predeclared Dice and extra-instance promotion checks failed.
- Experiment 006 blended experiment-003 and experiment-005 probabilities. It
  improved all tracked OOF instance diagnostics slightly, but its verified
  public score tied experiment 003 at 0.66.
- Experiment 007 tested boundary-aware three-head U-Nets and seeded
  partitioning; its grouped OOF gate failed, so it was not submitted.
- Experiment 008 is a controlled capacity/resolution ablation: 1024×1024 with
  32 base channels versus 1280×1280 with 24, using fixed targets and
  post-processing. Its validated 1024-base32 candidate scored 0.66 publicly,
  tying experiments 003 and 006.
- Experiment 009 changes only the optimization budget: it continues the
  selected 1024/base32 checkpoints with a lower learning rate and fixed
  post-processing.
- Experiment 010 calibrates only post-processing for the fixed Experiment 008
  probabilities using a 144-candidate grouped-OOF grid.
- Experiment 011 screens multi-scale flip TTA variants on the fixed
  Experiment-008 checkpoints, with no additional training or test tuning.
- Experiment 001 has a verified public score of 0.52. No leaderboard rank is
  claimed.

## Competition essentials

- Each prediction is one filament instance mask.
- Training annotations are COCO-style polygons with bounding boxes, spines, and
  chirality categories; the competition target is the segmentation only.
- Submission columns are `filament_id` and `segmentation_rle`.
- Masks use COCO compressed RLE counts for a fixed size of 2048 × 2048 pixels.
- Final judging is 70% quantitative and 30% qualitative. Quantitative review
  includes Dice, IoU, and fragmentation/over-segmentation distributions.
- The public metric currently rewards empty or severely under-predicted
  submissions. The organizer says the leaderboard is only a preliminary filter;
  this project therefore selects models with grouped OOF instance diagnostics,
  not by exploiting the public score.
- At most five submissions may be made per day and two may be selected as final.
- The solution repository must be public at competition close for final
  evaluation.

See [docs/competition.md](docs/competition.md) for the verified rules, timeline,
data schema, and source links.
See [docs/project-history.md](docs/project-history.md) for the safe command log,
experiment results, Kaggle run/submission references, and raw-log locations.
See [docs/experiment-004-operations.md](docs/experiment-004-operations.md) for
the exact private-kernel push, status, download, validation, and approved
submission commands for experiment 004.
See [docs/experiment-005-operations.md](docs/experiment-005-operations.md) for
the experiment-005 private-kernel and artifact-validation workflow.
See [docs/experiment-006-operations.md](docs/experiment-006-operations.md) for
the reviewed experiment-006 push, status, download, validation, and submission
commands.
See [docs/experiment-007-operations.md](docs/experiment-007-operations.md) for
the boundary-aware private-kernel and artifact-validation workflow.
See [docs/experiment-008-operations.md](docs/experiment-008-operations.md) for
the capacity/resolution private-kernel and artifact-validation workflow.
See [docs/experiment-009-operations.md](docs/experiment-009-operations.md) for
the low-learning-rate continuation workflow.
See [docs/experiment-010-operations.md](docs/experiment-010-operations.md) for
the OOF post-processing calibration workflow.
See [docs/experiment-011-operations.md](docs/experiment-011-operations.md) for
the multi-scale TTA workflow.

## Local setup

Use PowerShell from this repository:

```powershell
.\scripts\setup.ps1
.\scripts\verify_kaggle_access.ps1
.\scripts\download_competition_data.ps1
.\.venv\Scripts\python.exe .\scripts\inspect_data.py
.\scripts\validate.ps1
```

The Kaggle CLI reads the credential already configured outside this repository.
These scripts do not create, display, copy, or modify a token.

The extracted data layout is:

```text
data/raw/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

Competition data, credentials, trained weights, generated metrics, predictions,
and submissions are ignored by Git.

## Project layout

```text
configs/                         Versioned experiment settings
data/README.md                   Local and Kaggle data locations
docs/                            Rules, validation policy, experiment registry
kaggle/                          Self-contained private-kernel source
outputs/README.md                Generated-output conventions
scripts/                         Setup, access, download, and validation commands
src/solar_filament_segmentation/ Reusable data, metric, RLE, and submission code
tests/                           Synthetic tests with no competition data
```

## Validation policy

Identical physical JPEG observations can have independent annotations from
different annotators. Validation must group by the underlying `file_name`, not
only by annotation-set `image["id"]`, so the same pixels never appear in both
training and validation.

Model selection, threshold tuning, connected-component filtering, and
post-processing are performed only with fixed grouped validation. Test images
are used only for final inference. The low-level Dice/IoU helpers in this repo
are diagnostics; they do not pretend to reproduce the organizer's complete
instance-matching evaluation.

## Development

```powershell
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

Every real experiment gets a unique configuration and output directory. Record
its decision in [docs/experiments.md](docs/experiments.md), while keeping the
generated artifacts out of Git.
