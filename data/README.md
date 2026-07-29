# Competition data

This directory is intentionally documentation-only in Git. All downloaded data
is ignored.

After running `scripts/download_competition_data.ps1`, the expected local root is:

```text
data/raw/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                                   # 707 JPEG files
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  # COCO-style annotations
└── test/
    └── test_images/                                    # 180 JPEG files
```

The Kaggle notebook input root is expected to be:

```text
/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

The downloaded archive is 670.98 MiB and its 888 extracted files total
716.18 MiB. The verified annotation inventory is:

- 1,154 COCO image records representing 707 physical observations;
- 411 observations annotated once, 145 twice, and 151 three times;
- 8,199 filament annotations;
- 2,535 Left, 2,590 Right, and 3,074 Unidentifiable annotations;
- an Ambiguous category defined in the schema but unused in this training split.

All images are 2048 × 2048 grayscale H-alpha observations stored as 8-bit JPEG.
The setup audit found no exact observation-stem or SHA-256 overlap between train
and test and no byte-identical duplicate files within either split. The generated
audit is stored at `outputs/setup/data-audit.json` and is ignored by Git.

Do not commit the archive, extracted images, or annotations. Competition data is
licensed for the non-commercial competition, academic-research, and education
uses described by the official rules.
