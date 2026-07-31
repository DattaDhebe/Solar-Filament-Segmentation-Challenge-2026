# Experiment 006 operations

Experiment 006 reuses the five experiment-003 and five experiment-005
competition-trained checkpoints. It performs no additional fitting. It
reconstructs grouped OOF probabilities, evaluates 135 predeclared probability
blend and component-filter settings, checks that both source candidates
reproduce their recorded OOF results, freezes the selected setting, and only
then loads test observations for ensemble inference.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-006-oof-blend-calibration.yaml
Get-Content kaggle\kernel-metadata.json
```

The private metadata must use:

```text
kernel: dattadhebe/solar-filament-oof-blend-calibration
sources:
  dattadhebe/solar-filament-oof-tta-refinement/1
  dattadhebe/solar-filament-hybrid-annotator/1
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
  dattadhebe/solar-filament-oof-blend-calibration

.\.venv\Scripts\kaggle.exe quota --format json
```

After status becomes `COMPLETE`:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-oof-blend-calibration/1 `
  -p outputs\kaggle\experiment-006-v1
```

If Kaggle stops after OOF selection but before final inference, validate the
interruption-safe OOF artifact:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_006_output.py `
  outputs\kaggle\experiment-006-v1 `
  --oof-only
```

For a completed run, validate the OOF provenance, promotion gate, metadata/CSV
agreement, test identifiers, and every canonical COCO RLE:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_006_output.py `
  outputs\kaggle\experiment-006-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

If the strict validator reports a failed promotion gate, retain the run as
experiment evidence and do not submit it as the priority candidate.

## Submit only after completed validation and separate approval

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-006-v1\experiment-006-submission.csv `
  -m "experiment-006 OOF-selected experiment-003/005 probability blend"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```
