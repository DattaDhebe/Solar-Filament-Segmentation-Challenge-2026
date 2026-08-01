# Project execution history

This is the durable, human-readable command and decision log for the Solar
Filament Segmentation Challenge 2026 entry. It records material commands,
validation results, Kaggle runs, submissions, experiment diagnostics, rule
checks, and Git commits through 2026-07-30.

## Safety and scope

- This file provides a detailed engineering decision journal: observations,
  alternatives, decisions, evidence, commands, and outcomes. It does not contain
  private hidden chain-of-thought or unrecorded internal reasoning.
- Commands are shown relative to the repository root.
- Kaggle is used only through the configured CLI as an opaque authenticated
  client. Credential files and token values are never read or logged.
- Competition data, model weights, raw metrics, predictions, masks, kernel
  metadata, and submission CSVs remain Git-ignored.
- Long raw Kaggle logs and generated metadata are referenced by local path
  rather than copied into Git.
- Kaggle kernel pushes and competition submissions were performed only after
  explicit user approval.
- Kaggle submission timestamps below are the timestamps returned by the CLI.
- Local Dice and IoU values are diagnostics, not claims of exact organizer
  evaluator parity.

## Current verified state

| Experiment | Kaggle kernel | Submission | Public score |
| --- | --- | --- | ---: |
| 001 | `dattadhebe/solar-filament-first-submission`, version 1 | `55089873` | 0.52 |
| 002 | `dattadhebe/solar-filament-overnight-ensemble`, version 1 | `55099193` | 0.62 |
| 003 | `dattadhebe/solar-filament-oof-tta-refinement`, version 1 | `55107032` | 0.66 |

Latest submission-status command:

```powershell
.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 --page-size 10
```

Verified output on 2026-07-30:

```text
55107032  experiment-003-submission.csv  COMPLETE  0.66
55099193  experiment-002-submission.csv  COMPLETE  0.62
55089873  submission.csv                 COMPLETE  0.52
```

No private score or leaderboard rank is claimed.

## Repository initialization and data audit

The workspace was initialized with fixed competition invariants, safe ignore
rules, data/schema helpers, RLE validation, submission checks, and synthetic
tests.

Material commands:

```powershell
.\scripts\setup.ps1
.\scripts\verify_kaggle_access.ps1
.\scripts\download_competition_data.ps1
.\.venv\Scripts\python.exe .\scripts\inspect_data.py
.\scripts\validate.ps1
```

Verified inventory:

```text
COCO image records: 1,154
Physical annotated observations: 707
Duplicate annotator image records: 447
Filament annotations: 8,199
Training JPEGs: 707
Test JPEGs: 180
Image size: 2048 x 2048
Image mode: grayscale L
Train/test shared stems: 0
Train/test shared exact hashes: 0
```

Validation grouping is always the physical observation `file_name`. The fixed
five-fold assignment fingerprint is:

```text
69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7
```

Generated data audit:

```text
outputs/setup/data-audit.json
```

## Experiment 001: first end-to-end baseline

Versioned configuration:

```text
configs/experiment-001-first-submission.yaml
```

Design:

- one fixed grouped validation fold;
- 512 x 512 grayscale input;
- small U-Net trained from scratch;
- no external labels or pretrained weights;
- semantic foreground prediction;
- connected-component instance separation;
- canonical 2048 x 2048 COCO compressed RLE output.

Local validation:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
.\scripts\validate.ps1
```

Kernel metadata was initialized without exposing credentials:

```powershell
.\.venv\Scripts\kaggle.exe kernels init -p kaggle
```

Private GPU launch and status:

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4

.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-first-submission
```

Kaggle reported:

```text
Kernel version 1 successfully pushed.
KernelWorkerStatus.RUNNING
KernelWorkerStatus.COMPLETE
```

Artifact download:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-first-submission/1 `
  -p outputs\kaggle\experiment-001-v1
```

Run results:

| Diagnostic | Result |
| --- | ---: |
| Runtime | 300.403 seconds |
| Best grouped validation semantic Dice | 0.640299 |
| Submission instances | 1,970 |
| Test images with predictions | 177 of 180 |
| RLE validation | 1,970 canonical masks passed |

Additional mechanical checks:

```text
Header: filament_id,segmentation_rle
Empty RLE values: 0
Duplicate filament IDs: 0
Mask-area pixels, min/median/max: 64 / 576 / 17,504
Instances per image, min/median/max: 0 / 11 / 29
Tracebacks: 0
NaNs: 0
CUDA out-of-memory errors: 0
```

Submission allowance and upload:

```powershell
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-001-v1\submission.csv `
  -m "experiment-001 first submission small U-Net fold-0 baseline"
```

The upload reached 100%, after which the CLI returned:

```text
Expecting value: line 1 column 1 (char 0)
```

The command was not retried blindly. A read-only submission-list check verified
that Kaggle had accepted reference `55089873`. Its final public score is 0.52.

Raw local artifacts:

```text
outputs/kaggle/experiment-001-v1/experiment-001-run-metadata.json
outputs/kaggle/experiment-001-v1/solar-filament-first-submission.log
outputs/kaggle/experiment-001-v1/submission.csv
outputs/kaggle/experiment-001-v1/experiment-001-model.pt
```

## Experiment 002: five-fold 1024 ensemble

Versioned configuration:

```text
configs/experiment-002-overnight-ensemble.yaml
```

Design:

- five immutable grouped folds;
- 1024 x 1024 grayscale input;
- five small U-Nets with 24 base channels, trained from scratch;
- up to 24 epochs per fold with early stopping;
- geometric and intensity augmentation;
- five-model probability ensemble;
- 27 post-processing candidates selected only from grouped OOF predictions;
- local greedy instance matching with penalties for missed and extra instances.

### Experiment-002 decision journal

This is the engineering rationale that led from experiment 001 to experiment
002. It is reconstructed from the versioned configuration, source diff, run
metadata, and Kaggle log.

| Observation or constraint | Decision | Why | Evidence or outcome |
| --- | --- | --- | --- |
| Experiment 001 used only fold 0 and scored 0.52 publicly. | Train all five fixed folds. | A single fold leaves most labeled observations unused for model fitting and provides a high-variance estimate. Five grouped models use every observation for training in four folds and validation in one fold. | All five folds completed; mean fold semantic Dice was 0.648139. |
| Multiple annotation-set IDs can refer to the same JPEG. | Continue grouping by physical `file_name`, never annotation ID. | Splitting duplicate annotation records across train and validation would leak the same solar observation. | The fold fingerprint remained `69a31d...5aa7` and all 1,154 annotation records stayed grouped by 707 observations. |
| Thin filaments lose detail at the 512 resolution used by experiment 001. | Increase model input to 1024 x 1024. | This preserves four times as many input pixels while remaining feasible on a T4 with a small U-Net and batch size 2. | The five-fold T4 run completed without an out-of-memory error. |
| The user had an overnight Kaggle GPU window. | Allow 24 epochs per fold, require at least 10, and use patience-5 early stopping. | This makes productive use of the time window while allowing a plateaued fold to stop. | Best epochs were 23, 10, 24, 21, and 22; runtime was 4.553 hours. |
| The competition wording restricts other ground-truth metadata. | Train from scratch with no external labels and no pretrained weights. | This is the conservative rule-compliant choice and avoids hidden-label leakage from public MAGFiLO annotations. | Both safeguards are explicit in YAML and covered by tests. |
| One semantic model is sensitive to fold-specific variation. | Average probability maps from all five fold models. | A probability ensemble generally reduces fold variance before instance separation. | Test inference used all five saved checkpoints and produced predictions for 179 of 180 images. |
| Semantic Dice alone does not measure split/merged instances. | Generate instances with connected components and evaluate grouped OOF instance diagnostics. | The submission requires individual filament instances, so semantic foreground cannot be the final output. | The log records matched Dice, matched IoU, missed/extra instances, and one-to-many/many-to-one behavior. |
| The experiment-001 post-processing values were not justified across all folds. | Search 27 combinations of threshold, closing kernel, and minimum area on OOF predictions only. | Post-processing must be selected without looking at test images or leaderboard response. | OOF selected threshold 0.60, closing 5, and minimum area 32. |
| Predictions included many small fragments. | Use a penalized diagnostic that includes both missed and extra predictions. | Matched-only Dice can look good while ignoring false components. Penalizing both sides makes fragmentation visible. | Matched Dice was 0.704663, but penalized Dice was only 0.430385 and extra instances totaled 4,432. This directly motivated experiment 003. |

Rejected or deferred alternatives:

- Full 2048 x 2048 training was not selected because five-fold training at that
  resolution would substantially increase T4 memory use and runtime before the
  1024 baseline was understood.
- External MAGFiLO labels were rejected because they include hidden test
  observations and the organizer prohibited using that additional ground truth.
- Supervised pretrained weights were deferred under the same conservative
  interpretation of the rules.
- Leaderboard-driven threshold tuning was rejected because it would tune on
  hidden test feedback and the public metric was known to reward empty masks.
- Collapsing all foreground into one mask was rejected because the task and
  submission format require filament instances.

### Experiment-002 command record

The material commands were:

```powershell
# Review the experiment configuration and shared kernel source.
Get-Content configs\experiment-002-overnight-ensemble.yaml
Get-Content kaggle\first_submission.py

# Run formatting, Ruff, synthetic tests, and the repository/data safety audit.
.\scripts\validate.ps1
git diff --check

# After explicit user approval, launch the private T4 kernel.
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# Check startup and final state without changing the run.
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-overnight-ensemble

# Download the completed, ignored artifacts.
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-overnight-ensemble/1 `
  -p outputs\kaggle\experiment-002-v1

# Inspect the run record and training log.
Get-Content `
  outputs\kaggle\experiment-002-v1\experiment-002-run-metadata.json
Get-Content `
  outputs\kaggle\experiment-002-v1\solar-filament-overnight-ensemble.log

# Re-run repository validation and validate every candidate RLE as described
# in "Full submission-validation command pattern" below.
.\scripts\validate.ps1

# After artifact review and explicit submission approval, submit the candidate.
.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-002-v1\experiment-002-submission.csv `
  -m "experiment-002 five-fold 1024 U-Net ensemble OOF-selected postprocessing"

# Verify acceptance and score instead of inferring either from the submit call.
.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 --page-size 10
```

Some read-only inspections were issued as combined PowerShell commands during
the interactive session. They are written one per line above so the workflow is
reproducible. Credential-file commands are deliberately absent because no
credential file was read.

### Files changed for experiment 002 and why

| File or generated location | Change | Reason |
| --- | --- | --- |
| `configs/experiment-002-overnight-ensemble.yaml` | Added the complete experiment-002 configuration. | Makes folds, seeds, resolution, model size, training budget, loss, augmentation, post-processing grid, metrics, and output schema reviewable and reproducible. |
| `kaggle/first_submission.py` | Expanded the experiment-001 script into a five-fold trainer with OOF probability collection, instance diagnostics, grid selection, five-model test ensembling, run metadata, and strict submission writing. | One self-contained script is directly executable by a private Kaggle script kernel without relying on a package install or internet access. |
| `kaggle/kernel-metadata.json` | Temporarily pointed at the private `solar-filament-overnight-ensemble` T4 kernel with competition data and internet disabled. | Kaggle uses this ignored control file to choose the private kernel, accelerator, and inputs. It is not committed because it contains account-specific operational metadata. |
| `tests/test_kaggle_kernel.py` | Extended synthetic checks for full grouped-fold coverage and disabled external assets. | Prevents a future edit from silently dropping folds or enabling prohibited data/model sources. |
| `README.md` | Added the experiment-002 design, diagnostics, verified score, and public-metric warning. | Keeps the repository overview accurate without claiming rank or private score. |
| `docs/experiments.md` | Added experiment 002 to the concise experiment registry. | Records the promotion decision and the remaining over-segmentation problem. |
| `docs/external-data.md` | Strengthened the MAGFiLO leakage warning and organizer-rule interpretation. | Explains why apparently available public labels were not used. |
| `CHANGELOG.md` | Recorded the five-fold ensemble and OOF-selection capability. | Provides a compact repository-level history. |
| `outputs/kaggle/experiment-002-v1/` | Generated five weights, raw Kaggle log, metadata, and the candidate CSV. | These are necessary run artifacts for review and reproducibility checks but are ignored and never committed. |

Git provenance note: the experiment-002 and experiment-003 source work was
committed together in `74a00ba` (`feat: add five-fold refinement pipeline`).
The experiment-002 YAML is preserved, but the shared
`kaggle/first_submission.py` was then evolved in place into the experiment-003
entry script. Therefore, the current Git tree does not pretend to be a separate
byte-for-byte snapshot of the Python script pushed for experiment 002. The raw
experiment-002 Kaggle log and metadata remain in the ignored output directory.

Private GPU launch and checks:

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-overnight-ensemble
```

Kaggle reported version 1 as `RUNNING` on three startup checks and later
`COMPLETE`.

Artifact download:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-overnight-ensemble/1 `
  -p outputs\kaggle\experiment-002-v1
```

Run results:

| Diagnostic | Result |
| --- | ---: |
| Runtime | 4.553 hours |
| Mean best-fold semantic Dice | 0.648139 |
| Penalized OOF instance Dice | 0.430385 |
| Mean matched-instance Dice | 0.704663 |
| Mean matched-instance IoU | 0.563493 |
| Matched instances | 7,166 |
| Missed instances | 1,033 |
| Extra instances | 4,432 |
| One-to-many relations | 651 |
| Many-to-one relations | 126 |
| Submission instances | 1,771 |

Fold results:

| Fold | Best epoch | Best semantic Dice |
| ---: | ---: | ---: |
| 0 | 23 | 0.653758 |
| 1 | 10 | 0.641641 |
| 2 | 24 | 0.643388 |
| 3 | 21 | 0.652332 |
| 4 | 22 | 0.649573 |

OOF-selected processing:

```text
probability threshold: 0.60
closing kernel: 5
minimum component area at model resolution: 32
```

Submission validation:

```text
1,771 canonical 2048 x 2048 RLE masks passed
Known test images: 180
Images with predictions: 179
Empty RLE values: 0
Duplicate filament IDs: 0
Mask-area pixels, min/median/max: 128 / 828 / 17,452
Instances per image, min/median/max: 0 / 10 / 23
Tracebacks: 0
NaNs: 0
CUDA out-of-memory errors: 0
```

Submission command:

```powershell
.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-002-v1\experiment-002-submission.csv `
  -m "experiment-002 five-fold 1024 U-Net ensemble OOF-selected postprocessing"
```

Kaggle accepted reference `55099193`. Its final public score is 0.62.

Raw local artifacts:

```text
outputs/kaggle/experiment-002-v1/experiment-002-run-metadata.json
outputs/kaggle/experiment-002-v1/solar-filament-overnight-ensemble.log
outputs/kaggle/experiment-002-v1/experiment-002-submission.csv
outputs/kaggle/experiment-002-v1/experiment-002-fold-0.pt
outputs/kaggle/experiment-002-v1/experiment-002-fold-1.pt
outputs/kaggle/experiment-002-v1/experiment-002-fold-2.pt
outputs/kaggle/experiment-002-v1/experiment-002-fold-3.pt
outputs/kaggle/experiment-002-v1/experiment-002-fold-4.pt
```

## Competition-rule and leaderboard investigation

Authenticated official file inventory:

```powershell
.\.venv\Scripts\kaggle.exe competitions files `
  filament-segmentation-2026 --page-size 200 --format json
```

The official competition source contains only:

```text
707 training JPEGs
180 test JPEGs
1 training annotation JSON
```

There is no second official labeled dataset.

Authenticated evaluation and rules:

```powershell
.\.venv\Scripts\kaggle.exe competitions pages `
  filament-segmentation-2026 list filament-segmentation-2026 `
  --content --page-name evaluation --format json

.\.venv\Scripts\kaggle.exe competitions pages `
  filament-segmentation-2026 list filament-segmentation-2026 `
  --content --page-name rules --format json
```

Authenticated leaderboard and discussion checks:

```powershell
.\.venv\Scripts\kaggle.exe competitions leaderboard `
  filament-segmentation-2026 --show --page-size 20

.\.venv\Scripts\kaggle.exe competitions topics list `
  filament-segmentation-2026 --format json

.\.venv\Scripts\kaggle.exe competitions topics show 728474 --format json
.\.venv\Scripts\kaggle.exe competitions topics show 729809 --format json
.\.venv\Scripts\kaggle.exe competitions topics show 729396 --format json
.\.venv\Scripts\kaggle.exe competitions topics show 728062 --format json
```

Verified findings:

- Generic external data is allowed only when public and reasonably accessible.
- Inference may use only the supplied H-alpha test images.
- Models may not use other ground-truth metadata for training.
- The organizer clarified that MAGFiLO ground truth must be limited to
  competition-provided annotations; otherwise the submission is disregarded.
- All 180 test filenames reportedly have labels in the public MAGFiLO release,
  so those public labels would leak every hidden target.
- Participants demonstrated that empty or nearly empty submissions can score
  above 0.9 and even 1.0 because of a public-metric defect.
- The organizer states that the public leaderboard is only a preliminary
  filter and that segmentation quality, reproducibility, and the full
  quantitative/qualitative rubric determine awards.

Decision:

```text
Do not use public MAGFiLO annotations.
Do not exploit the empty-mask score.
Use official labels only.
Select model and post-processing changes with grouped OOF diagnostics.
```

A failed inventory attempt used unsupported `--json`; it was corrected to
`--format json`. Failed page-command ordering attempts were corrected to the
working syntax recorded above.

## Experiment 003: consensus fine-tuning, TTA, and refinement

Versioned configuration:

```text
configs/experiment-003-oof-tta-refinement.yaml
```

Design:

- resume all five experiment-002 competition-trained checkpoints;
- one soft foreground target per physical training observation, averaging its
  independent annotator masks;
- preserve each annotator record separately for grouped OOF evaluation;
- up to 18 additional epochs per fold at learning rate 0.0002;
- early stopping after at least eight epochs;
- four-way test-time augmentation;
- five-model ensemble;
- 72 validation-only post-processing candidates;
- component area and mean-confidence filtering;
- official competition labels only.

### Experiment-003 decision journal

This is the engineering rationale that led from experiment 002 to experiment
003.

| Observation or constraint | Decision | Why | Evidence or outcome |
| --- | --- | --- | --- |
| Experiment 002 improved the public score from 0.52 to 0.62, but its penalized OOF instance Dice was 0.430385. | Keep the five-fold foundation and target the instance errors rather than replacing the whole pipeline. | The semantic model had useful signal; the larger gap was between matched quality and penalized instance quality. | Experiment 003 increased penalized OOF Dice to 0.490129 and public score to 0.66. |
| Experiment 002 produced 4,432 extra instances, far more than its 1,033 misses. | Add stronger area filtering and component mean-confidence filtering. | The error profile indicated fragmentation and low-confidence false components. | Extra instances fell to 2,103 and test rows fell from 1,771 to 1,332. |
| The same physical JPEG can have masks from multiple annotators. | Average its official annotator foreground masks into a soft consensus training target. | Treating duplicate annotations as unrelated hard targets can give contradictory supervision. A soft target preserves disagreement without leaking between folds. | Fine-tuning used 707 physical-observation targets while evaluation retained all 1,154 official annotation records. |
| Five experiment-002 checkpoints already contained competition-only learned features. | Resume each matching fold checkpoint instead of restarting. | The user asked for more training, and continuing from rule-compliant checkpoints is more time-efficient than repeating from random initialization. | Every fold log records its `experiment-002-fold-N.pt` source. |
| Fine-tuning from an established checkpoint can overwrite useful features. | Reduce learning rate from 0.001 to 0.0002, use at most 18 epochs, minimum 8, and patience 4. | A smaller update is appropriate for refinement and fits the available overnight GPU window. | Runtime was 2.883 hours; best fine-tuning epochs ranged from 5 to 12. |
| Predictions may vary under flips even though the physical target should transform consistently. | Average identity, horizontal-flip, vertical-flip, and combined-flip predictions. | Four-way TTA reduces orientation-specific prediction variance using only the supplied image. | Mean semantic Dice improved from 0.651170 before TTA to 0.652722 with TTA. |
| Experiment 002 searched only 27 post-processing settings and could not filter a component by confidence. | Search 72 OOF-only combinations, adding minimum mean probability 0.0/0.7/0.8 and larger minimum areas. | The expanded grid directly tests ways to suppress extra fragments without using test observations. | Selected threshold 0.50, closing 7, area 96, and mean confidence 0.80. |
| Stronger filtering may trade false positives for false negatives. | Report missed and extra counts separately instead of presenting only the improved aggregate. | The tradeoff matters for scientific review and future tuning. | Extras improved by 2,329, while misses increased by 660. The log retains both facts. |
| Public 1.00 scores were associated with a metric defect. | Continue selecting by grouped OOF instance quality, not by deleting predictions to imitate the exploit. | An empty-mask strategy would not be a legitimate segmentation improvement and could fail final qualitative/reproducibility review. | Experiment 003 produced 1,332 real, validated filament masks; no empty RLE rows were submitted. |

Rejected or deferred alternatives:

- Restarting five models from scratch was rejected because experiment-002
  checkpoints were already rule-compliant and the user specifically wanted more
  productive training.
- Training on public MAGFiLO ground truth was rejected because it contains the
  hidden test targets and violates the organizer clarification.
- Selecting the confidence threshold from test-image prediction counts or the
  leaderboard was rejected to preserve strict train/test separation.
- More aggressive filtering was not chosen merely to reduce row count; the
  selected setting had to win the grouped OOF penalized diagnostic.
- A different large architecture or supervised encoder was deferred because it
  would change several variables at once and could violate the conservative
  pretrained-weight policy.

### Experiment-003 command record

The material commands were:

```powershell
# Review the new configuration, checkpoint source, and implementation.
Get-Content configs\experiment-003-oof-tta-refinement.yaml
Get-Content kaggle\first_submission.py
Get-Content kaggle\kernel-metadata.json

# Validate before requesting a push.
.\scripts\validate.ps1
git diff --check

# After explicit user approval, launch the private refinement kernel.
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# Verify startup and eventual completion.
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-oof-tta-refinement

# Download artifacts into a distinct ignored directory.
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-oof-tta-refinement/1 `
  -p outputs\kaggle\experiment-003-v1

# Inspect recorded folds, checkpoints, TTA metrics, grid results, and errors.
Get-Content `
  outputs\kaggle\experiment-003-v1\experiment-003-run-metadata.json
Get-Content `
  outputs\kaggle\experiment-003-v1\solar-filament-oof-tta-refinement.log

# Re-run the repository gate and fully decode/canonicalize all 1,332 RLEs.
.\scripts\validate.ps1

# After human review and explicit user approval, submit experiment 003.
.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-003-v1\experiment-003-submission.csv `
  -m "experiment-003 consensus fine-tuning TTA five-fold OOF instance refinement"

# Confirm acceptance and the eventual public score.
.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 --page-size 10

# Review exactly what entered the source commit.
git status --short
git diff --check
git add CHANGELOG.md README.md `
  configs/experiment-002-overnight-ensemble.yaml `
  configs/experiment-003-oof-tta-refinement.yaml `
  docs/experiments.md docs/external-data.md `
  kaggle/first_submission.py tests/test_kaggle_kernel.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add five-fold refinement pipeline"
git status --short
```

The commit staged an explicit source/documentation list. It did not stage the
ignored metadata, checkpoints, logs, predictions, masks, metrics, or submission
CSV.

### Files changed for experiment 003 and why

| File or generated location | Change | Reason |
| --- | --- | --- |
| `configs/experiment-003-oof-tta-refinement.yaml` | Added checkpoint fine-tuning mode, checkpoint root, soft-consensus target strategy, lower learning rate, TTA list, and 72-candidate post-processing grid. | Keeps every material experiment-003 choice declarative, reviewable, and reproducible. |
| `kaggle/first_submission.py` | Evolved the shared script to resolve only experiment-002 fold checkpoints, construct observation-level consensus targets, fine-tune each fold, invert/average TTA transforms, filter instances by area and mean confidence, compare all OOF candidates, and record expanded metadata. | Implements the refinement hypothesis while maintaining grouped validation, grayscale loading, instance output, and strict RLE checks. |
| `kaggle/kernel-metadata.json` | Changed the private kernel identity to `solar-filament-oof-tta-refinement` and added `dattadhebe/solar-filament-overnight-ensemble/1` as a kernel source; internet remained disabled. | Makes the experiment-002 checkpoints available read-only inside Kaggle without external downloads. This account-specific file remains ignored. |
| `tests/test_kaggle_kernel.py` | Pointed the embedded-config parity test at experiment 003 and added assertions for refinement mode, experiment-002-only checkpoint provenance, consensus targets, and all five folds. | Detects configuration drift and accidental use of an unapproved checkpoint source. |
| `README.md` | Documented experiment 003, the experiment-002 baseline, and the decision not to exploit the public metric. It was later updated with the completed 0.66 score. | Gives reviewers an accurate current overview and separates verified facts from pending runs. |
| `docs/experiments.md` | Added experiment 003 and later replaced “running” with its completed diagnostics and verified score. | Maintains the compact experiment comparison and promotion record. |
| `docs/external-data.md` | Added the organizer clarification, test-label leakage warning, and empty-mask metric warning. | Records why external public annotations and score exploits were rejected. |
| `CHANGELOG.md` | Recorded consensus fine-tuning, TTA, confidence filtering, and the expanded OOF search. | Makes the material capability change visible at repository level. |
| `outputs/kaggle/experiment-003-v1/` | Generated five fine-tuned weights, raw log, run metadata, and the 1,332-row candidate CSV. | Preserves the evidence required for review while keeping generated artifacts out of Git. |

No files under `src/solar_filament_segmentation/` were changed in commit
`74a00ba`. The Kaggle job remained deliberately self-contained in
`kaggle/first_submission.py`; reusable local validation helpers under `src/`
continued to validate the downloaded candidate.

Private kernel source configuration:

```text
competition source: filament-segmentation-2026
kernel source: dattadhebe/solar-filament-overnight-ensemble/1
internet: disabled
GPU: NVIDIA T4
```

Launch and status:

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-oof-tta-refinement
```

Immediate and delayed startup checks reported `KernelWorkerStatus.RUNNING`.
The final check reported `KernelWorkerStatus.COMPLETE`.

Artifact download:

```powershell
.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-oof-tta-refinement/1 `
  -p outputs\kaggle\experiment-003-v1
```

Experiment 003 completed successfully and improved the grouped OOF instance
diagnostics over experiment 002.

| Metric | Experiment 002 | Experiment 003 |
| --- | ---: | ---: |
| Mean semantic Dice | 0.6481 | 0.6527 with TTA |
| Penalized OOF instance Dice | 0.4304 | 0.4901 |
| Matched-instance Dice | 0.7047 | 0.7215 |
| Matched-instance IoU | 0.5635 | 0.5819 |
| Missed instances | 1,033 | 1,693 |
| Extra instances | 4,432 | 2,103 |
| Test predictions | 1,771 | 1,332 |

Additional checks:

```text
Runtime: 2.883 hours
Mean best fine-tuning semantic Dice: 0.651170
Mean TTA semantic Dice: 0.652722
Matched instances: 6,506
One-to-many relations: 510
Many-to-one relations: 117
Tracebacks: 0
Errors: 0
NaNs: 0
CUDA out-of-memory errors: 0
```

Fold results:

| Fold | Best fine-tuning epoch | Fine-tuning Dice | TTA Dice |
| ---: | ---: | ---: | ---: |
| 0 | 12 | 0.657749 | 0.659203 |
| 1 | 10 | 0.655008 | 0.657890 |
| 2 | 12 | 0.646336 | 0.648174 |
| 3 | 5 | 0.652026 | 0.650399 |
| 4 | 6 | 0.644728 | 0.647945 |

OOF-selected processing:

```text
probability threshold: 0.50
closing kernel: 7
minimum component area at model resolution: 96
minimum component mean confidence: 0.80
```

Submission validation:

```text
1,332 canonical 2048 x 2048 RLE masks passed
Known test images: 180
Images with predictions: 176
Empty RLE values: 0
Duplicate filament IDs: 0
Mask-area pixels, min/median/max: 384 / 1,182 / 17,680
Instances per image, min/median/max: 0 / 8 / 18
Zero-prediction images: 4
```

Submission command:

```powershell
.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f outputs\kaggle\experiment-003-v1\experiment-003-submission.csv `
  -m "experiment-003 consensus fine-tuning TTA five-fold OOF instance refinement"
```

Kaggle accepted reference `55107032`. Its final public score is 0.66.

Raw local artifacts:

```text
outputs/kaggle/experiment-003-v1/experiment-003-run-metadata.json
outputs/kaggle/experiment-003-v1/solar-filament-oof-tta-refinement.log
outputs/kaggle/experiment-003-v1/experiment-003-submission.csv
outputs/kaggle/experiment-003-v1/experiment-003-fold-0.pt
outputs/kaggle/experiment-003-v1/experiment-003-fold-1.pt
outputs/kaggle/experiment-003-v1/experiment-003-fold-2.pt
outputs/kaggle/experiment-003-v1/experiment-003-fold-3.pt
outputs/kaggle/experiment-003-v1/experiment-003-fold-4.pt
```

## Future experiment roadmap

These are engineering suggestions, not promised leaderboard gains. Public score
1.00 is not the optimization target because the known empty-mask defect can
produce that score without valid segmentation. Every promotion must be justified
by grouped OOF instance diagnostics and qualitative mask review before a
competition submission is considered.

### Recommended order

| Priority | Proposed ID | Main hypothesis | Compute estimate | Promotion gate |
| ---: | --- | --- | --- | --- |
| 1 | 004 | An OOF-trained component-quality ranker and fragment-linking pass can reduce experiment 003's 2,103 extra instances without a large recall loss. | 2-4 T4 hours, mostly checkpoint inference | Penalized OOF Dice above 0.505, at least 15% fewer extras, and no more than 5% more misses than 003 |
| 2 | 005 | Hybrid consensus/annotator supervision can recover filaments lost by the stronger experiment-003 filtering. | 3-6 T4 hours for a screen; 6-10 hours for five-fold confirmation | Penalized OOF Dice above the promoted 004 result and lower misses without restoring 003-level extras |
| 3 | 006 | A boundary-aware multi-head U-Net can separate touching filaments better than connected components of one semantic channel. | 8-12 T4 hours | Better penalized Dice plus lower one-to-many and many-to-one rates on the same folds |
| 4 | 007 | More spatial detail or model capacity can improve thin-filament recall after the instance pipeline is reliable. | 8-12 T4 hours per confirmed setting | At least +0.005 mean OOF semantic Dice and a positive instance-Dice change |
| 5 | 008 | Self-supervised pretraining on official training JPEGs can improve feature quality without external labels or test leakage. | 8-12 hours pretraining plus supervised confirmation | Consistent gain across at least four of five folds and improved instance metrics |

The gates are intentionally stricter than “public score increased.” They can be
revised before implementation, but must be fixed before examining test
predictions or leaderboard feedback.

### Experiment 004: cross-fitted component quality and fragment linking

Why this is first:

- Experiment 003 already has reasonable matched-instance quality
  (`0.721510` Dice), but still produces 2,103 unmatched extra instances.
- Improving the decision about which connected components to keep is cheaper
  and more directly supported by the error data than immediately training a
  much larger model.
- It can reuse the five experiment-003 checkpoints and official labels only.

Implementation:

1. Preserve experiment 003 unchanged. Create:

   ```text
   configs/experiment-004-component-quality.yaml
   kaggle/experiment_004.py
   tests/test_experiment_004.py
   ```

   Use a new script instead of overwriting `first_submission.py`, so the exact
   source for every future Kaggle run remains versioned.

2. Load each experiment-003 fold checkpoint and recompute probability maps only
   for that fold's grouped validation observations. Never generate a training
   feature row from a model that trained on the same `file_name`.

3. From every candidate component, compute features that do not require hidden
   metadata:

   ```text
   area
   perimeter
   mean/max/standard-deviation probability
   probability quantiles
   major/minor axis lengths
   eccentricity
   solidity
   bounding-box aspect ratio
   skeleton length
   estimated width = area / skeleton length
   distance from solar-disk center
   normalized radial position
   distance to the eroded disk boundary
   nearest-component distance
   nearest-component tangent alignment
   ```

4. Derive a component quality target from official OOF instances. For a physical
   observation with multiple annotators, compute each candidate's maximum IoU
   to an instance in each annotation set, then average those maxima. This
   produces an annotator-aware continuous quality target without copying one
   image across training and validation.

5. Train a small deterministic model such as
   `HistGradientBoostingRegressor` on four OOF feature folds and predict the
   fifth. Repeat for all five folds. This outer cross-fitting is mandatory:
   evaluating a ranker on the same components used to fit it would give an
   optimistic result.

6. Search the component-score cutoff using only the concatenated cross-fitted
   predictions. Report score calibration, per-fold diagnostics, per-image Dice
   distributions, missed/extra counts, and split/merge behavior.

7. Add a conservative fragment-linking candidate grid. Connect two components
   only when their skeleton endpoints are close, endpoint tangents align, the
   probability along the gap is sufficiently high, and the merged shape remains
   filament-like. Select link distance, alignment, and gap-probability settings
   on the same cross-fitted OOF predictions.

8. For test inference, fit the ranker once on all OOF component rows, then apply
   it to experiment-003 ensemble components. Do not fit normalization,
   calibration, thresholds, or feature selection to test components.

Suggested configuration additions:

```yaml
component_quality:
  model: histogram-gradient-boosting-regressor
  seed: 20260730
  max_depth: 3
  max_iter: 150
  learning_rate: 0.05
  quality_cutoffs: [0.10, 0.15, 0.20, 0.25, 0.30]
  cross_fit_by: validation_fold

fragment_linking:
  enabled_candidates: [false, true]
  maximum_endpoint_distance: [0, 8, 16, 24]
  minimum_tangent_cosine: [0.80, 0.90]
  minimum_gap_probability: [0.30, 0.40, 0.50]
```

Keep the actual grid modest after a synthetic timing test. The values above are
starting hypotheses, not OOF-selected results.

Required tests:

- feature extraction is deterministic;
- feature values contain no NaNs or infinities;
- cross-fitting never trains on the predicted row's `file_name`;
- duplicate annotator records remain in one fold;
- linking never crosses the solar-disk boundary;
- each output instance is non-empty and unique;
- RLE remains canonical at 2048 x 2048;
- embedded and versioned YAML configurations match.

### Experiment 005: hybrid annotator-consensus supervision

Why:

- Soft consensus helped reduce false positives, but experiment 003 misses rose
  from 1,033 to 1,693.
- Averaging annotators can lower the target probability of filaments marked by
  only one annotator. A hybrid loss can retain consensus while exposing the
  model to valid annotator-specific structures.

Implementation:

1. Add:

   ```text
   configs/experiment-005-hybrid-annotator.yaml
   kaggle/experiment_005.py
   tests/test_experiment_005.py
   ```

2. Keep the immutable five grouped folds and load only experiment-003
   competition-trained checkpoints.

3. For every physical training observation, return both:

   - the soft mean of all official annotator masks;
   - one deterministic, epoch-seeded annotator mask.

4. Optimize a weighted hybrid objective:

   ```text
   total =
       consensus_weight * BCE+Dice(soft_consensus)
       + annotator_weight * BCE+Dice(sampled_annotator)
   ```

   Screen fixed weight pairs such as `(0.75, 0.25)` and `(0.50, 0.50)`.
   Do not choose weights from test behavior.

5. Use a two-stage budget:

   - screen the two fixed weight pairs on two predeclared grouped folds;
   - choose one pair using only those folds;
   - run the selected pair on all five folds and report the full OOF result.

6. Feed its OOF probabilities through the already fixed experiment-004
   component pipeline. If post-processing is retuned, label the result as a
   separate experiment variant so model and post-processing effects remain
   distinguishable.

Suggested configuration:

```yaml
training:
  target_strategy: hybrid-consensus-random-annotator
  consensus_loss_weight: 0.75
  annotator_loss_weight: 0.25
  learning_rate: 0.0001
  epochs: 12
  minimum_epochs: 6
  early_stopping_patience: 3
```

Required additional reporting:

- recall and missed instances by number of annotators available;
- component precision and extra instances;
- Dice distribution for single-annotator versus multi-annotator observations;
- the exact screen folds and decision made before full confirmation.

### Experiment 007: boundary-aware instance model

Why:

- Connected components can merge touching filaments and split faint,
  interrupted filaments.
- Official COCO annotations provide instance masks, so foreground and boundaries
  can be learned without external data.

Implementation:

1. Extend the U-Net decoder to three one-channel heads:

   ```text
   foreground probability
   instance-boundary probability
   centerline or signed-distance target
   ```

2. Build targets from official instance masks at model resolution:

   - union mask for foreground;
   - per-instance morphological boundary, clipped so adjacent objects remain
     separable;
   - normalized distance transform or skeleton heatmap inside each instance.

3. Use a weighted multi-task loss with values fixed in YAML. Start with:

   ```yaml
   loss:
     foreground: 1.0
     boundary: 0.5
     distance_or_centerline: 0.25
   ```

4. At inference, threshold foreground, suppress high boundary probability,
   generate seeds from centerline/distance peaks, run marker-controlled
   watershed, and regrow instances inside the foreground mask.

5. Tune only a compact threshold/seed grid on grouped OOF predictions. Compare
   directly with connected components using identical fold probabilities.

6. Report one-to-many and many-to-one counts as primary diagnostics for this
   experiment. A semantic Dice increase alone is insufficient to promote it.

Start by initializing the shared encoder/decoder from experiment 003 and adding
new randomly initialized heads. If optimization is unstable, run a clean
from-scratch control on one predeclared fold before spending on all five.

### Experiment 008: resolution and capacity ablation

Do this only after experiment 004 or 006 improves instance handling; otherwise a
larger semantic model may simply generate more fragments.

Implementation:

1. Compare one variable at a time on the same two predeclared screen folds:

   ```text
   A: 1024 input, 32 base channels
   B: 1280 input, 24 base channels
   C: 1536 input, 16 or 24 base channels
   ```

2. Use batch size 1 and gradient accumulation 2 for larger inputs. Keep effective
   batch size, seed, augmentations, targets, folds, and post-processing fixed.

3. Run a short synthetic/T4 memory probe before the real job. Abort a setting
   after a repeatable out-of-memory condition rather than silently reducing
   resolution or changing architecture.

4. Promote one setting to all five folds only if it gains at least 0.005
   semantic Dice on both screen folds and does not worsen the instance
   diagnostic.

5. Record throughput, peak GPU memory when available, runtime per epoch, and
   total cost in metadata.

Multi-scale TTA can be tested afterward as a separate inference-only variant.
Do not combine capacity, resolution, new augmentation, and new post-processing
in one run because the source of any gain would be unknowable.

### Experiment 009: official-training-only self-supervised pretraining

This is lower priority and must use only the 707 official training JPEGs.
Competition test images must remain inference-only and cannot be included in
self-supervised training.

Implementation:

1. Train the existing encoder as a masked autoencoder or denoising autoencoder
   using fixed grayscale crops from official training images.
2. Save a checkpoint whose metadata contains the exact list/hash of training
   filenames and asserts that no test filename was present.
3. Initialize the supervised segmentation encoder from that checkpoint; all
   decoder and output heads remain task-trained on official labels.
4. Compare against the identical randomly initialized architecture on the same
   grouped folds.
5. Promote only if gains are consistent across at least four folds and instance
   diagnostics improve, not merely training loss.

This approach uses no supervised external weights, but organizer rules should be
rechecked immediately before implementation in case they change.

### Implementation workflow for each future experiment

Use this sequence and preserve a separate source snapshot:

```powershell
# 1. Create a unique YAML and script, then add synthetic tests.
# Example names only; do not overwrite an earlier experiment script.
Get-Content configs\experiment-004-component-quality.yaml
Get-Content kaggle\experiment_004.py

# 2. Run the full local gate.
.\scripts\validate.ps1
git diff --check
git status --short

# 3. Review the exact source diff and ensure ignored artifacts are absent.
git diff -- configs kaggle src tests docs README.md CHANGELOG.md

# 4. Obtain explicit user approval, then push one private GPU kernel.
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# 5. Monitor, download to a unique ignored directory, and validate.
.\.venv\Scripts\kaggle.exe kernels status <owner/new-kernel>
.\.venv\Scripts\kaggle.exe kernels output <owner/new-kernel/version> `
  -p outputs\kaggle\<new-experiment-version>
.\scripts\validate.ps1

# 6. Compare against the fixed experiment-003/004 OOF baselines.
# Do not inspect or tune against test-image statistics.

# 7. Only after artifact validation and human review, request separate
# explicit approval for a competition submission.
```

For all future experiments:

- freeze the hypothesis and promotion gate in YAML before training;
- preserve the same physical-observation folds and fingerprint;
- save OOF diagnostics by fold and physical observation;
- record per-image distributions, not only means;
- report both improvements and regressions;
- never use test images for pseudo-label training, normalization fitting,
  calibration, threshold selection, feature selection, or stopping decisions;
- never use external MAGFiLO annotations;
- never optimize toward the empty-mask public-score exploit;
- never overwrite an earlier experiment's versioned source script;
- keep Kaggle credentials, kernel metadata, weights, logs, predictions, masks,
  metrics, and submission CSVs out of Git.

## How the pipeline was implemented

### Configuration and invariant enforcement

Every experiment has a versioned YAML file under `configs/`. The loader in
`src/solar_filament_segmentation/config.py` rejects configurations that violate
competition invariants:

- competition slug must be `filament-segmentation-2026`;
- masks must be 2048 x 2048;
- observations must be grayscale;
- validation group key must be `file_name`;
- task must remain instance segmentation;
- external labeled data must be disabled;
- submission columns and RLE format must be exact.

The Kaggle API uploads one code file, so the active YAML is embedded in
`kaggle/first_submission.py`. `tests/test_kaggle_kernel.py` parses the Python
source with `ast`, loads the embedded YAML, and requires exact equality with the
versioned config. This prevents a pushed kernel from silently diverging from
the reviewed experiment.

### COCO loading and physical-observation grouping

`load_examples()` reads only the official training JSON. It associates every
annotation polygon with its COCO image record and derives the physical
observation key from `Path(file_name).stem`.

The grouped split algorithm:

1. sorts the 707 unique physical observation keys;
2. shuffles them with the fixed validation seed `20260729`;
3. assigns them round-robin across five folds;
4. maps every annotation-set record for the same JPEG to the same fold;
5. serializes sorted `observation,fold` pairs;
6. hashes that canonical text with SHA-256.

This produced the immutable fingerprint:

```text
69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7
```

The test split is not read by split construction, fitting, threshold selection,
or model selection.

### Polygon rasterization

All official polygons are converted with `pycocotools`:

1. coordinates are scaled from 2048 to the configured model resolution;
2. `mask_utils.frPyObjects()` creates COCO RLE objects;
3. `mask_utils.decode()` rasterizes every instance;
4. the per-instance stack is combined into a semantic foreground mask;
5. a separate integer label map preserves the instance IDs for OOF matching.

The model always receives one grayscale channel. Images use bilinear resizing;
discrete masks use polygon rasterization at the target model resolution.

### U-Net architecture

The self-contained `SmallUNet` has:

- one input channel;
- base width 16 in experiment 001 and 24 in experiments 002/003;
- three encoder levels;
- a bottleneck at eight times the base width;
- max-pooling downsampling;
- transposed-convolution upsampling;
- skip concatenations;
- two convolution, batch-normalization, and ReLU stages per block;
- one-logit semantic foreground output.

Weights were initialized from scratch for experiments 001/002. Experiment 003
resumed only the five experiment-002 checkpoints trained from official
competition labels.

### Training targets

Experiments 001/002 treated every annotation-set record as a separate sample.
This meant the same JPEG could appear two or three times with independently
drawn masks.

Experiment 003 reduced that label noise with
`consensus_training_samples()`:

1. group annotation sets by physical observation;
2. verify that repeated records refer to identical cached pixels;
3. sum their semantic foreground masks;
4. divide by the number of annotators;
5. train once on the resulting soft target.

The held-out fold still keeps every annotator mask separate. Consensus is used
only for the training portion, so validation does not leak into fitting and
annotator disagreement remains visible in OOF diagnostics.

### Augmentation, optimizer, and loss

Training augmentation is deterministic under each fold seed and includes:

- horizontal flip;
- vertical flip;
- 90-degree rotations;
- bounded brightness shift;
- bounded contrast scaling.

The loss is:

```text
total loss = BCE weight * weighted binary cross entropy
           + Dice weight * soft Dice loss
```

The positive-class BCE weight is 4.0. Optimizer and scheduling:

```text
optimizer: AdamW
weight decay: 0.0001
experiment-002 starting learning rate: 0.001
experiment-003 fine-tuning learning rate: 0.0002
scheduler: cosine annealing
mixed precision: enabled on CUDA
```

The best checkpoint is selected only by semantic Dice on that fold's grouped
validation records. Early stopping cannot occur before the configured minimum
epoch count.

### Test-time augmentation and ensembling

Experiment 003 applies four transforms:

```text
identity
horizontal flip
vertical flip
horizontal + vertical flip
```

Each transformed prediction is flipped back to the original orientation before
averaging. The same TTA is used for OOF predictions and test inference. The five
fold-model probability maps are then averaged. Test masks or test-derived
statistics do not influence this averaging.

### Solar-disk restriction and instance separation

For each model-resolution image:

1. Otsu thresholding identifies bright solar content;
2. connected-component statistics select the largest disk region;
3. an elliptical erosion removes the bright limb;
4. the ensemble probability is thresholded;
5. morphological closing repairs small gaps;
6. 8-connected components create candidate instances;
7. candidates are filtered by area;
8. experiment 003 additionally filters by mean component probability;
9. remaining instances are sorted deterministically by area and centroid.

The pipeline is semantic during training but remains instance-aware at output
through this documented separation stage.

### OOF instance diagnostic and post-processing selection

For every held-out annotation set:

1. predicted components are compared with the integer ground-truth label map;
2. pairwise intersections, unions, Dice, and IoU are computed;
3. pairs below the configured minimum IoU are discarded;
4. remaining pairs are greedily matched by descending quality;
5. matched Dice and IoU are accumulated;
6. missed and extra instances are counted;
7. one-to-many and many-to-one overlap behavior is counted.

The penalized diagnostic is:

```text
sum of matched Dice / max(number of predicted instances,
                          number of target instances,
                          1)
```

This intentionally penalizes both missed and extra instances. It is a local
diagnostic, not a claim of organizer parity.

Experiment 002 evaluated 27 validation-only candidates across probability
threshold, closing kernel, and minimum area. Experiment 003 evaluated 72
candidates and added minimum mean component probability. The selected
parameters are applied to test predictions only after OOF selection is
finished.

### Submission encoding

Each selected model-resolution instance is resized to 2048 x 2048 with nearest
neighbor interpolation. `pycocotools.mask.encode()` receives a
Fortran-contiguous binary mask. Only its ASCII compressed `counts` string is
written.

Rows use:

```text
<test-image-stem>_<positive-instance-number>,<compressed-counts>
```

The CSV writer produces exactly:

```text
filament_id,segmentation_rle
```

### Kaggle packaging

`kaggle/kernel-metadata.json` is generated locally and ignored by Git. It sets:

- private kernel;
- Python script;
- NVIDIA T4;
- internet disabled;
- official competition source;
- experiment-002 kernel output as the sole checkpoint source for experiment
  003.

No Kaggle kernel performs a competition submission. Kernels only write
checkpoints, run metadata, logs, and a candidate CSV to `/kaggle/working`.

## Chronological action ledger

### 2026-07-29: repository and competition audit

```text
Inspected repository layout and documentation.
Added a challenge overview to README.md.
Verified Kaggle access and competition entry through the configured CLI.
Downloaded the official archive into ignored local storage.
Audited COCO schema, images, polygons, categories, duplicates, and split overlap.
Confirmed 707 physical training images and 180 test images.
Confirmed grayscale 2048 x 2048 JPEGs.
Confirmed no exact filename or SHA-256 train/test overlap.
Ran Ruff and 23 synthetic tests.
```

### 2026-07-29/30: experiment 001

```text
Added experiment-001 YAML.
Added a self-contained 512 x 512 U-Net Kaggle script.
Created private kernel metadata.
Received post-validation approval.
Pushed kernel version 1.
Verified RUNNING twice.
Downloaded COMPLETE artifacts.
Validated all 1,970 RLE masks.
Received submission approval.
Uploaded candidate.
Observed a CLI response-parse error after 100% upload.
Checked submission list instead of retrying.
Confirmed reference 55089873.
Verified final public score 0.52.
```

### 2026-07-30: experiment 002

```text
Expanded the kernel to 1024 x 1024.
Added five grouped fold models.
Added early stopping and cosine scheduling.
Added instance label maps and local instance matching.
Added 27 OOF post-processing candidates.
Added five-model test ensemble.
Ran Ruff and 26 tests.
Received post-validation approval.
Pushed private T4 kernel version 1.
Verified RUNNING three times.
Downloaded COMPLETE artifacts.
Validated all 1,771 RLE masks.
Reviewed fold and OOF diagnostics.
Received submission approval.
Confirmed reference 55099193.
Verified final public score 0.62.
```

### 2026-07-30: rule and metric investigation

```text
Verified multiple live leaderboard entries at 1.00.
Queried the official competition file inventory.
Confirmed there is no second official labeled dataset.
Fetched authenticated Evaluation and Rules content.
Listed and read discussions about empty-mask scoring and public MAGFiLO overlap.
Confirmed public MAGFiLO includes all 180 hidden test labels.
Confirmed organizer prohibition on using external MAGFiLO ground truth.
Confirmed the public metric can reward empty predictions.
Decided not to download leaked labels or exploit the metric.
```

### 2026-07-30: experiment 003

```text
Added experiment-003 YAML.
Attached only our experiment-002 competition-trained checkpoints.
Added continued low-learning-rate training.
Added soft annotator-consensus training targets.
Added four-way TTA.
Added component confidence filtering.
Expanded OOF search to 72 candidates.
Ran Ruff and 27 tests.
Received post-validation approval.
Pushed private T4 kernel version 1.
Verified RUNNING immediately and after checkpoint mounting.
Downloaded COMPLETE artifacts.
Reviewed all fold histories and OOF candidate summaries.
Validated all 1,332 RLE masks.
Received submission approval.
Confirmed reference 55107032.
Verified final public score 0.66.
```

## Notable failures, gates, and corrections

### README encoding context

An initial patch matched PowerShell's incorrectly decoded display of the
multiplication sign and failed to apply. The file was reread explicitly as
UTF-8, and the patch was applied against the actual `×` character. No file
encoding was changed.

### Ruff invoked on YAML

One early command passed a YAML path to Ruff, which produced Python syntax
errors. The YAML itself was valid. Subsequent Ruff commands target Python files,
while YAML is loaded and checked through the project configuration tests.

### Required Kaggle push approvals

Initial private-kernel pushes were blocked when approval had not been renewed
after local validation. No workaround was attempted. For each experiment the
resulting sequence was:

```text
finish local implementation
run Ruff, tests, and data audit
report the reviewed configuration
receive explicit post-validation approval
push private Kaggle kernel
verify RUNNING
```

### Experiment-001 submit response

The CLI uploaded 100% of the CSV and then failed to parse an empty/non-JSON
response. The upload was not repeated. `competitions submissions` confirmed the
accepted reference, preventing a duplicate submission.

### Competition file-output option

`competitions files --json` was rejected because that subcommand uses:

```text
--format json
```

The corrected command is recorded in the rule-investigation section.

### Competition page-command syntax

The installed Kaggle CLI exposed the competition argument at both the parent
and list-subcommand levels. Initial single-position attempts returned “No
competition specified” or an invalid command choice. The working authenticated
syntax uses the slug in both accepted positional locations, as recorded above.

### Public web-page response

Unauthenticated page reads returned the Kaggle shell without rule content. The
authenticated Kaggle CLI `competitions pages` command was used instead. No
authentication data was inspected or copied.

### Metric exploit

The live leaderboard and discussions showed 1.00 scores from empty-mask
behavior. The project did not imitate this. Model decisions remain tied to
grouped OOF instance quality because the organizer says poor segmentation
strategies are not prize-worthy.

## Files changed by the work

Versioned source and documentation:

```text
README.md
CHANGELOG.md
configs/experiment-001-first-submission.yaml
configs/experiment-002-overnight-ensemble.yaml
configs/experiment-003-oof-tta-refinement.yaml
docs/competition.md
docs/experiments.md
docs/external-data.md
docs/project-history.md
docs/validation.md
kaggle/first_submission.py
src/solar_filament_segmentation/config.py
src/solar_filament_segmentation/data.py
src/solar_filament_segmentation/metrics.py
src/solar_filament_segmentation/rle.py
src/solar_filament_segmentation/submission.py
tests/test_config.py
tests/test_data.py
tests/test_kaggle_kernel.py
tests/test_metrics.py
tests/test_repository_safety.py
tests/test_rle.py
tests/test_submission.py
```

Generated and ignored:

```text
data/raw/
outputs/setup/
outputs/kaggle/experiment-001-v1/
outputs/kaggle/experiment-002-v1/
outputs/kaggle/experiment-003-v1/
kaggle/kernel-metadata.json
*.pt
submission*.csv
```

## Full submission-validation command pattern

Every candidate CSV was parsed with `csv.DictReader`, converted to
`SubmissionRow` objects, and checked with:

```python
validate_submission_rows(
    rows,
    known_image_ids=known_test_image_ids,
    height=2048,
    width=2048,
    decode_masks=True,
)
```

The checks enforce:

- exact two-column header;
- known test-image stems;
- unique and correctly suffixed filament IDs;
- non-empty ASCII compressed RLE strings;
- successful 2048 x 2048 decode;
- non-empty decoded masks;
- canonical encode/decode round trips;
- no duplicate masks within one image.

## Git history

```text
44cc0c7  Initialize Solar Filament Segmentation Challenge workspace
4b3d5cc  feat: add first Kaggle submission baseline
74a00ba  feat: add five-fold refinement pipeline
```

Commit commands used:

```powershell
git add <explicit reviewed file list>
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add first Kaggle submission baseline"
git commit -m "feat: add five-fold refinement pipeline"
git status --short
```

Generated data, weights, metrics, logs, predictions, kernel metadata, and
submission files were reviewed before each commit and were not staged.

## Standard continuation workflow

For every future experiment:

```powershell
# 1. Validate source and synthetic behavior.
.\scripts\validate.ps1
git diff --check

# 2. After explicit post-validation approval, push the private kernel.
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# 3. Verify startup and eventual completion.
.\.venv\Scripts\kaggle.exe kernels status <owner/kernel-slug>

# 4. Download into a new ignored directory.
.\.venv\Scripts\kaggle.exe kernels output <owner/kernel-slug/version> `
  -p outputs\kaggle\<experiment-version>

# 5. Inspect metadata/logs and fully validate every RLE.
.\scripts\validate.ps1

# 6. After human review and explicit approval, check allowance and submit.
.\.venv\Scripts\kaggle.exe competitions submission-limits `
  filament-segmentation-2026 --json

.\.venv\Scripts\kaggle.exe competitions submit `
  -c filament-segmentation-2026 `
  -f <validated-candidate.csv> `
  -m "<unique experiment description>"

# 7. Confirm acceptance instead of retrying blindly.
.\.venv\Scripts\kaggle.exe competitions submissions `
  -c filament-segmentation-2026 --page-size 10
```

Do not tune against test observations or the broken public metric. Promote a
candidate only when grouped OOF instance quality, morphology, stability, and
reproducibility support it.

## Experiment 004 preparation - 2026-07-30

Status: source prepared locally; no Kaggle kernel push or competition
submission has been performed for experiment 004.

The experiment follows the first item in the documented roadmap. It keeps the
five experiment-003 U-Net checkpoints frozen and targets the remaining instance
error profile with an OOF-trained component-quality model plus a conservative
fragment-linking screen. This is deliberately a checkpoint-inference and
stacking experiment rather than another segmentation-model fine-tuning run.

### Decision journal

| Evidence | Change | Reason |
| --- | --- | --- |
| Experiment 003 retained 2,103 unmatched extra instances. | Generate a permissive component pool and estimate a continuous component-quality score. | A learned OOF quality estimate can use shape and probability evidence jointly instead of relying on one area/confidence cutoff. |
| The same physical JPEG can have multiple annotators. | Average each candidate's maximum IoU across that observation's official annotation sets. | The target respects annotator variation while keeping all duplicate records in one grouped fold. |
| Training and evaluating a component ranker on the same rows would be optimistic. | Train the ranker on four OOF feature folds and predict the untouched fifth, repeated across all folds. | Every OOF selection score remains cross-fitted by immutable `file_name` fold. |
| Fragmentation remains part of final judging. | Screen optional endpoint-distance, tangent-alignment, and gap-probability linking. | The link is accepted only inside the estimated solar disk and selected using grouped OOF annotations. |
| Test observations are inference-only. | Freeze the OOF-selected quality cutoff and linking setting before loading final test predictions into the output stage. | No test statistic, feature selection, normalization fit, threshold, or post-processing choice can influence promotion. |

### Files prepared

| File | Purpose |
| --- | --- |
| `configs/experiment-004-component-quality.yaml` | Freezes checkpoint provenance, grouped folds/fingerprint, candidate pool, component features, ranker, linking grid, diagnostics, and promotion gate. |
| `kaggle/experiment_004.py` | Self-contained private-kernel source for OOF inference, annotator-aware targets, cross-fitting, OOF selection, frozen test inference, canonical RLE output, and run metadata. |
| `tests/test_experiment_004.py` | Checks embedded/versioned config parity, disabled training/external sources, deterministic finite features, cross-fit group isolation, sparse diagnostics, and disk-boundary-safe linking. |
| `scripts/validate_experiment_004_output.py` | Validates provenance, fold fingerprint, configuration hash, OOF evidence, promotion gate, metadata/CSV agreement, and every 2048-square RLE. |
| `docs/experiment-004-operations.md` | Stores the exact push, status, output-download, validation, allowance, submission, and confirmation commands. |
| `kaggle/kernel-metadata.json` | Locally points the private kernel at `experiment_004.py` and the experiment-003 v1 kernel output; it remains intentionally ignored by Git. |

### Commands used during preparation

```powershell
git status --short
rg --files
git log -5 --oneline

Get-Content configs\experiment-003-oof-tta-refinement.yaml
Get-Content configs\experiment-002-overnight-ensemble.yaml
Get-Content kaggle\first_submission.py
Get-Content tests\test_kaggle_kernel.py

.\.venv\Scripts\ruff.exe format kaggle\experiment_004.py
.\.venv\Scripts\ruff.exe check kaggle\experiment_004.py
.\.venv\Scripts\pytest.exe tests\test_experiment_004.py -q
.\scripts\validate.ps1
git diff --check
git status --short
```

The push, completion check, output validation, and submission commands are
versioned in `docs/experiment-004-operations.md`. A kernel push still requires
explicit approval after the completed local validation and source review. A
competition submission requires a second explicit approval after the generated
candidate and grouped-OOF evidence have been reviewed.

## Experiment 004 cancelled run - 2026-07-30

Kaggle run `dattadhebe/solar-filament-component-quality/1` ended with
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`. The user reported that they did not
cancel it. The Kaggle status API returned no failure message, GPU quota still
had 8.55 hours remaining, no newer account kernel was present, and the log
contained no Python, CUDA, memory, or timeout error.

The run completed all OOF work before Kaggle stopped it during final output
encoding at test observation 75 of 180:

```text
selected quality cutoff: 0.25
fragment linking: disabled
OOF penalized instance Dice: 0.501474
missed instances: 1,807
extra instances: 1,673
promotion gate: failed
```

Relative to experiment 003, extras fell by 430 but misses rose by 114. The
predeclared gate failed because Dice remained below 0.505 and misses increased
by more than 5%. No complete submission CSV or final metadata existed, so the
cancelled artifact was not eligible for submission and an identical rerun was
not recommended.

Read-only investigation commands:

```powershell
.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-component-quality

.\.venv\Scripts\kaggle.exe kernels output `
  dattadhebe/solar-filament-component-quality/1 `
  -p outputs\kaggle\experiment-004-v1-cancelled

.\.venv\Scripts\kaggle.exe kernels logs `
  dattadhebe/solar-filament-component-quality/1

.\.venv\Scripts\kaggle.exe quota --format json
.\.venv\Scripts\kaggle.exe kernels list `
  --user dattadhebe --sort-by dateRun --page-size 20
```

The cancelled log remains under ignored output storage:

```text
outputs/kaggle/experiment-004-v1-cancelled/
solar-filament-component-quality.log
```

## Experiment 005 preparation - 2026-07-31

Experiment 005 responds directly to the recall regression in experiments 003
and 004. It uses only the five experiment-003 competition-trained checkpoints
and the fixed grouped-fold fingerprint.

The kernel screens two predeclared hybrid loss pairs on folds 0 and 1:

```text
consensus 0.75 / annotator 0.25
consensus 0.50 / annotator 0.50
```

Each training observation supplies the soft consensus mask and one
deterministic epoch-seeded annotator mask. The pair with higher screen-fold OOF
penalized instance Dice is selected. Its fold-0/1 models are reused, and only
folds 2 through 4 require additional training, giving seven fold-training runs
instead of nine. All five selected models use the frozen experiment-003
post-processing settings so the supervision change is isolated.

Prepared files:

```text
configs/experiment-005-hybrid-annotator.yaml
kaggle/experiment_005.py
tests/test_experiment_005.py
scripts/validate_experiment_005_output.py
docs/experiment-005-operations.md
```

The kernel writes `experiment-005-oof-metadata.json` immediately after
five-fold confirmation and before test inference. This preserves the selection
decision and all OOF evidence if Kaggle interrupts the final stage.

Experiment 004 was committed before experiment-005 development:

```text
c9b0149  feat: add cross-fitted component quality experiment
```

Local experiment-005 validation:

```powershell
.\.venv\Scripts\ruff.exe format .
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe tests\test_experiment_005.py -q
.\scripts\validate.ps1
git diff --check
git status --short
```

The full gate passed with 38 synthetic tests plus the complete read-only data
audit: 707 train observations, 180 test observations, immutable grouped folds,
and no filename or exact-hash split overlap.

## Experiment 005 completed result - 2026-07-31

Kaggle kernel `dattadhebe/solar-filament-hybrid-annotator/1` completed without
Python, CUDA, memory, NaN, or output-encoding errors. It selected the
`consensus=0.75, annotator=0.25` hybrid pair and produced 1,386 canonical test
instance masks in 5,911.5 seconds (98.5 minutes).

The strict validator stopped because the predeclared promotion gate failed; it
did not indicate a corrupt candidate. Validation without the optional gate
requirement confirmed the complete CSV and every RLE.

| Metric | Experiment 003 | Experiment 005 | Decision evidence |
| --- | ---: | ---: | --- |
| Penalized OOF instance Dice | 0.490129 | 0.490488 | Only a marginal increase and below the 0.500 gate |
| Matched-instance Dice | 0.7215 | 0.7225 | Slight overlap improvement |
| Matched-instance IoU | 0.5819 | 0.5830 | Slight overlap improvement |
| Missed instances | 1,693 | 1,609 | Recall improved by 84 instances |
| Extra instances | 2,103 | 2,309 | False-positive/fragment count worsened by 206 |
| Test predictions | 1,332 | 1,386 | Mechanically valid; not used for selection |

Promotion checks were `false/true/false` for Dice, misses, and extras. The
experiment is retained as useful recall evidence and a checkpoint source, but
experiment 003 remains the promoted submitted candidate until a better grouped
OOF result is validated.

Commands used to diagnose and validate the result:

```powershell
.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_005_output.py `
  outputs\kaggle\experiment-005-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images `
  --require-promotion-gate

.\.venv\Scripts\python.exe `
  .\scripts\validate_experiment_005_output.py `
  outputs\kaggle\experiment-005-v1 `
  --test-image-directory `
  data\raw\MAGFiLO_1.0_Kaggle_2026\test\test_images
```

Evidence remains in ignored output storage:

```text
outputs/kaggle/experiment-005-v1/experiment-005-oof-metadata.json
outputs/kaggle/experiment-005-v1/experiment-005-run-metadata.json
outputs/kaggle/experiment-005-v1/experiment-005-submission.csv
outputs/kaggle/experiment-005-v1/solar-filament-hybrid-annotator.log
```

## Experiment 006 preparation - 2026-07-31

Experiment 006 uses experiment 005's recall gain while targeting its extra
components. It does not perform another checkpoint fine-tune. Instead, it
reconstructs OOF probability maps from both completed checkpoint families and
selects one probability blend plus stricter post-processing setting entirely
from the immutable grouped OOF observations.

### Decision journal

| Evidence | Change | Reason |
| --- | --- | --- |
| Experiment 003 has fewer extras; experiment 005 has fewer misses. | Search experiment-005 probability weights `0.0, 0.25, 0.5, 0.75, 1.0`. | A probability blend can retain complementary precision and recall before connected-component separation. |
| Experiment 005 exceeded the extra-instance baseline by 206. | Search thresholds `0.50, 0.525, 0.55`, component areas `96, 128, 160`, and component mean confidences `0.80, 0.825, 0.85`; keep the OOF-selected closing kernel fixed at 7. | The compact 135-candidate grid targets small or weak components without introducing unrelated changes. |
| A wrong source version could invalidate calibration. | Require the grid endpoints to reproduce experiments 003 and 005 within `1e-5` Dice and exact missed/extra counts. | Test inference is blocked if checkpoint provenance or deterministic OOF reconstruction changes. |
| Selecting unconstrained Dice can keep an unacceptable error tradeoff. | Prefer candidates with at most 1,692 misses and 2,103 extras, then maximize penalized instance Dice. | The selection directly preserves the recall improvement while restoring experiment-003 precision. |
| OOF tuning introduces selection optimism. | Keep the grid compact and require a separate promotion threshold of 0.495 plus both count constraints. | A candidate is promoted only when the gain is large enough to justify a new competition submission. |
| Test images are inference-only. | Write the complete OOF selection artifact before loading any test JPEG. | Test statistics cannot influence blend, filtering, or promotion. |

Prepared files:

```text
configs/experiment-006-oof-blend-calibration.yaml
kaggle/experiment_006.py
tests/test_experiment_006.py
scripts/validate_experiment_006_output.py
docs/experiment-006-operations.md
```

The ignored `kaggle/kernel-metadata.json` now points to private kernel
`dattadhebe/solar-filament-oof-blend-calibration` with GPU enabled, internet
disabled, and exactly these two upstream private-kernel output sources:

```text
dattadhebe/solar-filament-oof-tta-refinement/1
dattadhebe/solar-filament-hybrid-annotator/1
```

No Kaggle push or competition submission was performed during preparation.

Local experiment-006 validation completed successfully:

```powershell
.\.venv\Scripts\ruff.exe format `
  kaggle\experiment_006.py `
  scripts\validate_experiment_006_output.py `
  tests\test_experiment_006.py

.\.venv\Scripts\ruff.exe check `
  kaggle\experiment_006.py `
  scripts\validate_experiment_006_output.py `
  tests\test_experiment_006.py

.\.venv\Scripts\pytest.exe tests\test_experiment_006.py -q
.\scripts\validate.ps1
git diff --check
git status --short
```

Results:

```text
Ruff: all checks passed
Experiment-006 synthetic tests: 7 passed
Repository synthetic suite: 45 passed
Train/test audit: 707/180 grayscale JPEGs; zero stem or exact-hash overlap
Configuration SHA-256:
673e16d2a6f3a5ffc9dc3ff68c227fffbb2b97c3b80080f007137b3431263e99
```

The 135-candidate loop caches the Otsu-derived, eroded solar-disk mask once per
training observation. This is a runtime optimization only: it does not change
any probability, threshold, connected component, OOF diagnostic, or selection
rule.

## Experiment 006 Kaggle push - 2026-07-31

After explicit user approval, private kernel version 1 was pushed successfully:

```powershell
.\.venv\Scripts\kaggle.exe kernels push `
  -p kaggle `
  --accelerator NvidiaTeslaT4 `
  --timeout 43200

.\.venv\Scripts\kaggle.exe kernels status `
  dattadhebe/solar-filament-oof-blend-calibration
```

Verified immediately after the push:

```text
kernel: dattadhebe/solar-filament-oof-blend-calibration/1
visibility: private
GPU: NVIDIA T4 requested
internet: disabled
status: KernelWorkerStatus.RUNNING
```

This action created a Kaggle kernel version only. No competition submission,
leaderboard score, or rank is claimed.

## Experiment 006 completed result and submission - 2026-07-31

Private kernel version 1 completed in 6,270.9 seconds (104.5 minutes). Both
source-reference candidates reproduced their exact recorded grouped-OOF Dice,
missed, and extra counts before blend selection. Of 135 candidates, exactly one
met the predeclared missed/extra feasibility constraints.

Selected setting:

```text
experiment-003 probability weight: 0.75
experiment-005 probability weight: 0.25
foreground threshold: 0.50
closing kernel: 7
minimum component area: 96
minimum component mean probability: 0.80
```

| Metric | Experiment 003 | Experiment 006 |
| --- | ---: | ---: |
| Penalized OOF instance Dice | 0.490129 | 0.491750 |
| Matched-instance Dice | 0.721510 | 0.722413 |
| Matched-instance IoU | 0.581948 | 0.582996 |
| Missed instances | 1,693 | 1,681 |
| Extra instances | 2,103 | 2,095 |
| One-to-many relations | 510 | 514 |
| Many-to-one relations | 117 | 120 |
| Test masks | 1,332 | 1,341 |

The strict promotion gate failed only because Dice remained below the absolute
0.495 requirement. The candidate nevertheless improved Dice, matched overlap,
misses, and extras over experiment 003, passed canonical-RLE validation, and
was chosen for one exploratory submission after checking that five submissions
were allowed.

```text
submission reference: 55137658
file: experiment-006-submission.csv
status: SubmissionStatus.COMPLETE
verified public score: 0.66
private score: not available
```

The public score tied experiment 003. This is evidence that probability blends
and small component-filter changes are saturated; the next experiment must
change learned instance separation rather than repeat threshold calibration.

## Experiment 007 preparation - 2026-07-31

Experiment 007 implements the previously documented boundary-aware hypothesis,
renumbered from the earlier roadmap because experiment 006 became the blend
calibration run. It is a substantive model and instance-separation change.

### Decision journal

| Evidence | Change | Reason |
| --- | --- | --- |
| Experiments 003 and 006 both scored 0.66 despite a larger post-processing search. | Replace the one-channel semantic output with foreground, boundary, and normalized instance-distance heads. | The model must learn evidence that connected components cannot express. |
| Official annotations contain separate filament polygons. | Derive a per-instance morphological boundary and per-instance normalized distance transform from each official annotator mask. | These targets require no external labels and preserve annotator-specific instance structure. |
| Experiment 005 showed a small recall benefit from annotator-specific supervision. | Use the fixed 0.75 consensus / 0.25 deterministic annotator foreground loss; train boundary and distance heads against the same epoch-selected annotator. | The foreground head retains consensus stability while auxiliary heads see coherent instances. |
| The experiment-003 U-Net already provides useful solar-filament features. | Load its encoder, decoder, and semantic head exactly; allow only the new boundary and distance heads to be missing and randomly initialized. | This isolates the boundary-aware change and avoids external pretrained weights. |
| Learned seeds can over-split thin filaments or disappear. | Compare one connected-components control with 96 seeded settings spanning boundary threshold, distance threshold, seed closing, minimum seed area, instance area, and confidence. | The control quantifies whether the new heads genuinely improve instance separation. |
| A higher Dice candidate could worsen recall or false positives. | Prefer candidates with at most 1,681 misses and 2,095 extras, then maximize penalized instance Dice. | Selection must preserve experiment 006's balanced error profile. |
| Semantic Dice alone cannot justify this experiment. | Require seeded partitioning, Dice at least 0.500, and no regression in missed, extra, one-to-many, or many-to-one counts. | Promotion demands evidence that the learned instance heads improve actual instance behavior. |
| Test observations are inference-only. | Complete training and all 97 grouped-OOF evaluations, write OOF evidence, and only then load test JPEGs. | Test statistics cannot affect training, early stopping, method selection, or promotion. |

Prepared files:

```text
configs/experiment-007-boundary-seeded-unet.yaml
kaggle/experiment_007.py
tests/test_experiment_007.py
scripts/validate_experiment_007_output.py
docs/experiment-007-operations.md
```

The ignored `kaggle/kernel-metadata.json` points to private GPU kernel
`dattadhebe/solar-filament-boundary-seeded-u-net`, has internet disabled, and
uses only `dattadhebe/solar-filament-oof-tta-refinement/1` as an upstream
competition-trained checkpoint source. No experiment-007 kernel push or
competition submission was performed during preparation.

Local experiment-007 validation:

```powershell
.\.venv\Scripts\ruff.exe format .
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe tests\test_experiment_007.py -q
.\scripts\validate.ps1
git diff --check
git status --short
```

Results:

```text
Ruff: all checks passed
Experiment-007 synthetic tests: 9 passed
Repository synthetic suite: 54 passed
Train/test audit: 707/180 grayscale JPEGs; zero stem or exact-hash overlap
Configuration SHA-256:
ff6d24135e3406d71f1c47c33633e61a80f68e2aa713142f0d86e1884645856a
```

The local environment intentionally does not execute real model training or
OpenCV inference. Those remain private-Kaggle GPU operations. Local tests cover
configuration parity, source-weight isolation, candidate-grid cardinality,
deterministic annotator selection, confidence filtering, constrained OOF
selection, relation-aware promotion, and the OOF-before-test boundary.

## Experiment 007 Kaggle push - 2026-07-31

After explicit user approval, the private T4 kernel was pushed successfully.
Kaggle normalized the title to a `u-net` slug, so the verified version-1 kernel
identifier is:

```text
dattadhebe/solar-filament-boundary-seeded-u-net/1
```

Immediate status verification returned `KernelWorkerStatus.RUNNING`. The kernel
is private, GPU-enabled, internet-disabled, and uses only the experiment-003
private kernel output as its checkpoint source. No competition submission was
made.

## Experiment 008 preparation - 2026-08-01

Experiment 007 did not clear its instance-quality gate, so the next run is a
controlled capacity/resolution ablation rather than another post-processing
search. Two settings are screened on folds 0 and 1 with identical grouped
targets, augmentations, TTA, and Experiment-006 post-processing:

```text
1024-base32: 1024x1024 input, 32 base channels
1280-base24: 1280x1280 input, 24 base channels
```

Both settings initialize from scratch so resolution and model capacity are not
confounded by incompatible checkpoint shapes. The selected setting is reused
for folds 2, 3, and 4. Promotion requires a predeclared +0.005 semantic Dice
gain on each screen fold, at least 0.495 full OOF penalized instance Dice, and
no increase beyond experiment-006's 1,681 misses or 2,095 extras. OOF metadata
is written before any test image is loaded.

Prepared files:

```text
configs/experiment-008-capacity-resolution.yaml
kaggle/experiment_008.py
tests/test_experiment_008.py
scripts/validate_experiment_008_output.py
docs/experiment-008-operations.md
```

The ignored `kaggle/kernel-metadata.json` is prepared for private GPU kernel
`dattadhebe/solar-filament-capacity-resolution-ablation`, with internet disabled
and no upstream checkpoint source. No experiment-008 push or submission has
been performed during preparation.

Local experiment-008 validation completed before any push:

```powershell
.\.venv\Scripts\ruff.exe format kaggle\experiment_008.py scripts\validate_experiment_008_output.py tests\test_experiment_008.py
.\.venv\Scripts\ruff.exe check kaggle\experiment_008.py scripts\validate_experiment_008_output.py tests\test_experiment_008.py
.\.venv\Scripts\pytest.exe tests\test_experiment_008.py -q
.\scripts\validate.ps1
git diff --check
```

Results: Ruff passed; the focused Experiment-008 tests passed (6 tests); the
full synthetic suite passed (60 tests); the train/test audit found 707/180
grayscale JPEGs with zero stem or exact-hash overlap. Configuration SHA-256:

```text
61742bc77455d569b11dbc4ee04c39e6f6e0d69ec8be990e729fce2806399f6e
```

Real training and inference remain private-Kaggle GPU operations. At this
preparation point, no experiment-008 kernel push or competition submission had
been performed.

## Experiment 008 Kaggle result - 2026-08-01

The approved private kernel completed successfully as:

```text
dattadhebe/solar-filament-capacity-resolution-ablation/1
```

Downloaded artifacts passed the complete local validator:

```text
selected setting: 1024-base32
OOF penalized instance Dice: 0.495771
matched instance Dice: 0.724427
matched instance IoU: 0.583907
missed instances: 1,669
extra instances: 1,935
one-to-many: 406
many-to-one: 150
test masks: 1,293
runtime: 18,257 seconds (about 5.07 hours)
```

The full Dice, missed, and extra checks passed, but the required screen
semantic gate did not: 1024-base32 reached 0.658474 and 0.655585 semantic Dice
on folds 0 and 1, while the baselines plus the required margin were 0.664203
and 0.662890. The 1280-base24 setting was lower still (0.648923 and 0.650400).
Therefore `promotion_gate.passed` is false and no Experiment-008 competition
submission was made. Kaggle submission history still contains only experiments
001, 002, 003, and 006.
