# Experiment 007 operations

Experiment 007 fine-tunes five boundary-aware U-Nets from the five immutable
experiment-003 checkpoints. Each model predicts foreground, per-instance
boundary probability, and normalized interior distance. A 97-candidate grouped
OOF comparison selects either the connected-components control or learned
seeded partitioning before any test JPEG is loaded.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-007-boundary-seeded-unet.yaml
Get-Content kaggle\kernel-metadata.json
```

The private metadata must use:

```text
kernel: dattadhebe/solar-filament-boundary-seeded-u-net
source: dattadhebe/solar-filament-oof-tta-refinement/1
GPU: enabled
internet: disabled
```

## Push only after explicit approval

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle `
  --accelerator NvidiaTeslaT4 `
  --timeout 43200
```

## Check status and quota

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-boundary-seeded-u-net

.\.venv\Scripts\kaggle.exe quota --format json
```

After status becomes `COMPLETE`:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-boundary-seeded-u-net/1 `
  -p outputs\kaggle\experiment-007-v1
```

If Kaggle stops after OOF selection but before final inference, validate the
interruption-safe OOF artifact:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_007_output.py `
  outputs\kaggle\experiment-007-v1 `
  --oof-only
```

For a completed run, require the boundary-aware promotion gate and validate all
five checkpoints, metadata, test identifiers, CSV rows, and canonical COCO RLE
masks:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_007_output.py `
  outputs\kaggle\experiment-007-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

If the strict validator reports a failed gate, retain the run as experiment
evidence and inspect which Dice, missed/extra, or relation check failed before
considering a submission.

## Submit only after completed validation and separate approval

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-007-v1\experiment-007-submission.csv `
  -m "experiment-007 boundary-aware three-head U-Net seeded partition"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```
