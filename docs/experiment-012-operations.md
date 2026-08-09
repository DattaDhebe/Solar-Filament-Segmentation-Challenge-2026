# Experiment 012 operations

Experiment 012 trains a 5-fold grouped U-Net pipeline using solar disk CLAHE
contrast-enhancement preprocessing and tiled 1024×1024 patch training on native
2048×2048 observations.

All five grouped OOF folds are evaluated before test images are loaded.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-012-patch2048-clahe.yaml
Get-Content kaggle\kernel-metadata.json
```

Private metadata:

```text
kernel: dattadhebe/solar-filament-patch2048-clahe
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
  dattadhebe/solar-filament-patch2048-clahe

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-patch2048-clahe/1 `
  -p outputs\kaggle\experiment-012-v1
```

Require the predeclared promotion gate before considering submission:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_012_output.py `
  outputs\kaggle\experiment-012-v1 `
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
  -f outputs\kaggle\experiment-012-v1\experiment-012-submission.csv `
  -m "experiment-012 patch2048 CLAHE pipeline"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```

If the gate fails, retain the OOF evidence and do not submit automatically.
