# Experiment 014 Operations Guide

Experiment 014 trains a 5-fold grouped Deep Supervision Attention UNet++ with Focal-Tversky Loss and distance-transform Watershed instance separation at 1024×1024 input resolution.

All five grouped OOF folds are evaluated before test images are loaded.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-014-attention-unet.yaml
Get-Content kaggle\kernel-metadata.json
```

Private metadata:

```text
kernel: dattadhebe/solar-filament-attention-unetpp-watershed
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

## Check and validate

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-attention-unetpp-watershed

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-attention-unetpp-watershed `
  -p outputs\kaggle\experiment-014-v1
```

Require the predeclared promotion gate before considering submission:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_013_output.py `
  outputs\kaggle\experiment-014-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

Submit only if that strict validation succeeds, the candidate is reviewed, and separate user approval is given:

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-014-v1\experiment-014-submission.csv `
  -m "experiment-014 attention-unet watershed pipeline"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```

If the gate fails, retain the OOF evidence and do not submit automatically.
