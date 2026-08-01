# Experiment 011 operations

Experiment 011 screens three inference-only TTA variants using the fixed
Experiment-008 `1024-base32` checkpoints and fixed post-processing:

- baseline flips at scale `1.0`;
- scales `0.75 / 1.0 / 1.25`;
- scales `0.875 / 1.0 / 1.125`.

All five grouped OOF folds are evaluated before test images are loaded.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-011-multiscale-tta.yaml
Get-Content kaggle\kernel-metadata.json
```

Private metadata:

```text
kernel: dattadhebe/solar-filament-multiscale-tta
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

## Check and validate

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-multiscale-tta

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-multiscale-tta/1 `
  -p outputs\kaggle\experiment-011-v1

.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_011_output.py `
  outputs\kaggle\experiment-011-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images
```

Require the predeclared promotion gate before considering submission:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_011_output.py `
  outputs\kaggle\experiment-011-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

Submit only if that strict validation succeeds, the selected variant is
reviewed, and you give separate approval:

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-011-v1\experiment-011-submission.csv `
  -m "experiment-011 multiscale TTA"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```

If the gate fails, retain the OOF evidence and do not submit automatically.
