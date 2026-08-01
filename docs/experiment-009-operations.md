# Experiment 009 operations

Experiment 009 continues the selected Experiment-008 `1024-base32` models from
their private Kaggle checkpoints. It uses 14 additional epochs at learning rate
`0.00005`; targets, grouped folds, augmentations, TTA, and post-processing are
unchanged. OOF metadata is written before any test JPEG is loaded.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-009-low-lr-continuation.yaml
Get-Content kaggle\kernel-metadata.json
```

The private metadata should use:

```text
kernel: dattadhebe/solar-filament-low-lr-continuation
GPU: enabled
internet: disabled
kernel source: dattadhebe/solar-filament-capacity-resolution-ablation/1
```

## Push only after explicit approval

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle `
  --accelerator NvidiaTeslaT4 `
  --timeout 43200
```

## Check and download artifacts

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-low-lr-continuation

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-low-lr-continuation/1 `
  -p outputs\kaggle\experiment-009-v1
```

Validate a complete run:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_009_output.py `
  outputs\kaggle\experiment-009-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images
```

To require the predeclared improvement gate:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_009_output.py `
  outputs\kaggle\experiment-009-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

Do not submit automatically if the gate fails. Review the full OOF diagnostics
and obtain separate approval before using the candidate CSV.
