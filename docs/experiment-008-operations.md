# Experiment 008 operations

Experiment 008 screens two from-scratch capacity/resolution settings on grouped
folds 0 and 1, then confirms the selected setting on all five immutable folds:
`1024-base32` and `1280-base24`. Targets, augmentation, TTA, and the
Experiment-006 post-processing parameters remain fixed. Test JPEGs are loaded
only after OOF metadata is written.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-008-capacity-resolution.yaml
Get-Content kaggle\kernel-metadata.json
```

The private metadata should use the Kaggle-normalized slug:

```text
kernel: dattadhebe/solar-filament-capacity-resolution-ablation
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

## Check status and download artifacts

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-capacity-resolution-ablation

.\.venv\Scripts\kaggle.exe quota --format json

# After status is COMPLETE:
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-capacity-resolution-ablation/1 `
  -p outputs\kaggle\experiment-008-v1
```

If the run stops after OOF selection but before final inference, validate the
interruption-safe OOF artifact:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_008_output.py `
  outputs\kaggle\experiment-008-v1 `
  --oof-only
```

For a complete run, validate all five checkpoints, metadata, and canonical
2048-by-2048 COCO RLE masks. The strict gate is intentionally separate from
artifact validation:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_008_output.py `
  outputs\kaggle\experiment-008-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

If the gate fails, retain the evidence and inspect the setting-level semantic
screen result and full OOF missed/extra/relation diagnostics. Do not submit a
failed candidate automatically.

## Submit only after completed validation and separate approval

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-008-v1\experiment-008-submission.csv `
  -m "experiment-008 capacity-resolution ablation"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```
