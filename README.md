# Solar Filament Segmentation Challenge 2026

Active competition workspace for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The task is instance segmentation of solar filaments in 2048 × 2048 grayscale
GONG H-alpha observations.

## Current status

- Kaggle access and competition entry are verified through the configured CLI.
- The full official dataset is downloaded to Git-ignored local storage.
- All 707 training and 180 test images are verified as 2048 × 2048 grayscale
  JPEGs, with no exact filename or SHA-256 overlap between the splits.
- The COCO JSON contains 1,154 annotation-set image records for 707 physical
  observations and 8,199 filament annotations.
- The isolated Python environment, repository safeguards, and 23 synthetic tests
  are ready.
- No model experiment, Kaggle notebook push, leaderboard submission, rank, or
  score is claimed yet.

## Competition essentials

- Each prediction is one filament instance mask.
- Training annotations are COCO-style polygons with bounding boxes, spines, and
  chirality categories; the competition target is the segmentation only.
- Submission columns are `filament_id` and `segmentation_rle`.
- Masks use COCO compressed RLE counts for a fixed size of 2048 × 2048 pixels.
- Final judging is 70% quantitative and 30% qualitative. Quantitative review
  includes Dice, IoU, and fragmentation/over-segmentation distributions.
- At most five submissions may be made per day and two may be selected as final.
- The solution repository must be public at competition close for final
  evaluation.

See [docs/competition.md](docs/competition.md) for the verified rules, timeline,
data schema, and source links.

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
