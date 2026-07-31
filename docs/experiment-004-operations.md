# Experiment 004 operations

Experiment 004 uses the five frozen experiment-003 checkpoints. It does not
continue U-Net training. It performs grouped out-of-fold inference, trains a
cross-fitted component-quality regressor, selects component filtering and
conservative fragment-linking settings only from OOF annotations, and finally
creates a test submission candidate.

The predeclared promotion gate is:

- mean grouped-OOF penalized instance Dice at least `0.505`;
- at least 15% fewer extra instances than experiment 003;
- no more than 5% more missed instances than experiment 003.

The public score is not part of parameter selection.

## Before pushing

Run from the repository root in PowerShell:

```powershell
.\scripts\validate.ps1
git diff --check
git status --short

Get-Content configs\experiment-004-component-quality.yaml
Get-Content kaggle\kernel-metadata.json
```

Confirm that the private metadata names:

```text
dattadhebe/solar-filament-component-quality
```

and attaches only:

```text
dattadhebe/solar-filament-oof-tta-refinement/1
```

`kaggle/kernel-metadata.json` is intentionally ignored by Git because it is an
account-specific operational file.

## Push the private GPU kernel

Pushing requires explicit approval after the source review:

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle `
  --accelerator NvidiaTeslaT4 `
  --timeout 43200
```

This creates a private Kaggle kernel artifact. It does not make a competition
submission.

## Check training/inference status

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-component-quality
```

Repeat the status command later. Download output only after the status is
`COMPLETE`:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-component-quality/1 `
  -p outputs\kaggle\experiment-004-v1
```

If Kaggle reports a version other than `1`, replace `/1` and the destination
suffix with the actual version. Never download over an earlier experiment
directory.

## Validate the completed output

First inspect the OOF diagnostics, selected setting, promotion gate, runtime,
and errors:

```powershell
Get-Content `
  outputs\kaggle\experiment-004-v1\experiment-004-run-metadata.json

Get-ChildItem outputs\kaggle\experiment-004-v1
```

Then run the full source gate and decode every candidate RLE:

```powershell
.\scripts\validate.ps1

.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_004_output.py `
  outputs\kaggle\experiment-004-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate
```

The output validator checks checkpoint provenance, immutable fold fingerprint,
configuration hash, absence of test-based selection, OOF metrics, metadata/CSV
row agreement, known image IDs, duplicate masks, empty masks, and the canonical
2048 by 2048 COCO compressed-RLE round trip.

If the promotion gate fails, retain the artifact as an experiment result but do
not submit it as the next promoted candidate without a documented review.

## Submit after review and separate approval

Check the daily allowance:

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 `
  --json
```

After the downloaded candidate has passed validation, its grouped-OOF evidence
has been reviewed, and the user has explicitly approved the competition
submission:

```powershell
.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-004-v1\experiment-004-submission.csv `
  -m "experiment-004 cross-fitted component quality and conservative fragment linking"
```

Confirm acceptance once instead of retrying blindly:

```powershell
.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 `
  --page-size 10
```
