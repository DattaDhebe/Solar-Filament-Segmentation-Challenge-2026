# Changelog

## Unreleased

### Added

- Initialized the local Git repository for the Solar Filament Segmentation
  Challenge 2026.
- Documented the authenticated official competition pages, data inventory,
  evaluation rubric, submission format, timeline, and rules.
- Added safe Kaggle access and data-download scripts.
- Downloaded and structurally audited the complete official competition archive
  in Git-ignored storage: 707 training images, 180 test images, 1,154 COCO image
  records, and 8,199 filament annotations.
- Verified all images are 2048 × 2048 grayscale JPEGs and found no exact
  filename or SHA-256 overlap within or across train/test splits.
- Added YAML experiment configuration, COCO data helpers, Dice/IoU diagnostics,
  COCO compressed-RLE helpers, submission validation, and synthetic tests.
- Added repository-wide credential, data, model, output, and submission ignores.
- Added a self-contained, private Kaggle GPU baseline that trains a small U-Net
  from scratch, preserves grouped validation, separates connected-component
  instances, and generates a mechanically validated submission.
- Ran experiment 001 and recorded its train-only fold fingerprint and semantic
  validation diagnostic without treating it as the organizer's matching score.
- Added experiment 002: five grouped 1024 × 1024 models, OOF instance
  diagnostics, validation-selected post-processing, and ensemble inference.
- Added experiment 003: continued five-fold training with soft
  annotator-consensus targets, TTA, component-confidence filtering, and an
  expanded OOF post-processing search.
- Prepared experiment 004: frozen experiment-003 checkpoint inference,
  annotator-aware OOF component targets, fold-cross-fitted component-quality
  ranking, conservative fragment linking, a strict output validator, and an
  operational push/check/submission command guide.
- Prepared experiment 005: two-stage hybrid consensus/annotator fine-tuning,
  reuse of selected screen models, full five-fold OOF confirmation, an
  interruption-safe OOF metadata checkpoint, and strict output validation.
- Recorded experiment 005's completed grouped-OOF result and failed promotion
  gate: recall improved, but extra instances rose above experiment 003.
- Prepared experiment 006: deterministic reconstruction of experiment-003 and
  experiment-005 OOF probabilities, a 135-candidate blend/component-filter
  calibration grid, source-result regression checks, interruption-safe OOF
  evidence, and strict output validation.
- Documented the organizer's prohibition on external MAGFiLO ground truth and
  the known empty-mask public-metric failure.
- Added a durable safe command and experiment history with Kaggle run,
  submission, validation, rule-investigation, artifact, and Git references.
- Expanded the history with experiment-002/003 decision journals,
  file-by-file rationale, reproducible command records, and an implementation
  roadmap for experiments 004 through 008.
