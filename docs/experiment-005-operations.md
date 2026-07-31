# Experiment 005 operations

Experiment 005 targets experiment 003's missed filaments with hybrid
consensus/annotator supervision. It screens the predeclared `(0.75, 0.25)` and
`(0.50, 0.50)` loss pairs on grouped folds 0 and 1, reuses the selected screen
models, trains folds 2 through 4, and confirms the selected pair across all five
immutable folds. Test observations are loaded only after OOF selection and
confirmation.

## Local gate and review

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-005-hybrid-annotator.yaml
Get-Content kaggle\kernel-metadata.json
```

The private metadata must use:

```text
kernel: dattadhebe/solar-filament-hybrid-annotator
source: dattadhebe/solar-filament-oof-tta-refinement/1
internet: disabled
```

## Push after explicit approval

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle `
  --accelerator NvidiaTeslaT4 `
  --timeout 43200
```

## Check status and quota

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-hybrid-annotator

.\.venv\Scripts\kaggle.exe quota --format json
```

After status becomes `COMPLETE`:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-hybrid-annotator/1 `
  -p outputs\kaggle\experiment-005-v1
```

If Kaggle cancels after full OOF confirmation, download the available output
and validate the pre-test checkpoint:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_005_output.py `
  outputs\kaggle\experiment-005-v1 `
  --oof-only
```

For a completed run, validate every output and require the promotion gate:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_005_output.py `
  outputs\kaggle\experiment-005-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

## Submit only after completed validation and separate approval

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-005-v1\experiment-005-submission.csv `
  -m "experiment-005 hybrid consensus and annotator recall fine-tuning"

.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```
