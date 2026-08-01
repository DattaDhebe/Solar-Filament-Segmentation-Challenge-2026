# Experiment 010 operations

Experiment 010 reuses the five validated Experiment-008 `1024-base32`
checkpoints and calibrates only post-processing on grouped OOF predictions.
The frozen grid contains 144 combinations of probability threshold, closing
kernel, minimum component area, and minimum component confidence. Test images
are loaded only after candidate selection and OOF metadata are written.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-010-postprocessing-calibration.yaml
Get-Content kaggle\kernel-metadata.json
```

Private metadata:

```text
kernel: dattadhebe/solar-filament-postprocessing-calibration
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

## Check, download, and validate

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-postprocessing-calibration

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-postprocessing-calibration/1 `
  -p outputs\kaggle\experiment-010-v1

.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_010_output.py `
  outputs\kaggle\experiment-010-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images
```

Use `--oof-only` for an interruption-safe OOF artifact. Require the promotion
gate only after reviewing the candidate diagnostics; a failed gate must not be
submitted automatically.
