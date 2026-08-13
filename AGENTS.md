# Permanent project instructions

- Treat this repository as an active entry in the Kaggle Solar Filament
  Segmentation Challenge 2026. Never claim a leaderboard submission, rank, or
  score unless it has been explicitly verified.
- Preserve strict train/test separation. Test observations are for final
  inference only; never use them for training, validation, normalization fitting,
  feature selection, threshold tuning, component filtering, or model selection.
- Use fixed seeds and validation splits grouped by the underlying physical
  observation `file_name`. Annotation-set IDs are not sufficient because the same
  JPEG may have annotations from multiple annotators.
- Optimize and report grouped out-of-fold instance-segmentation diagnostics,
  centered on Dice. Also report IoU, per-image distributions, missed/extra
  instances, fragmentation, and one-to-many/many-to-one behavior.
- Treat local Dice and IoU helpers as diagnostics unless the organizer's complete
  matching implementation has been reproduced exactly.
- Keep important model, augmentation, validation, threshold, and post-processing
  parameters in YAML configuration files. Give every experiment a unique ID and
  output directory.
- Use `pycocotools` for COCO polygon/mask conversion and compressed-RLE submission
  encoding. Submission masks are fixed at 2048 by 2048 pixels.
- Load H-alpha JPEG observations as grayscale. Do not silently process them as RGB.
- Preserve filament instances in predictions; do not collapse the task into a
  single semantic foreground mask without a documented instance-separation step.
- External data and models must be public, equally accessible, and rule-compliant.
  The evaluation page says models may not use other ground-truth metadata for
  training, so do not use external labeled data or supervised pretrained weights
  until the organizer has clarified that wording.
- Competition data may be used only for the allowed non-commercial competition,
  academic-research, and education purposes.
- Never commit datasets, credentials, access tokens, archives, trained models,
  generated metrics, OOF/test predictions, masks, or submissions.
- Never read, print, copy, or modify Kaggle credential files. Use the configured
  Kaggle CLI as an opaque authenticated client.
- Do not push a Kaggle notebook or make a competition submission without explicit
  user approval after local validation and human review. A notebook push is not a
  competition submission.
- Keep real-data training and inference on Kaggle unless the user explicitly
  approves another compute environment. Local work should focus on source,
  configuration, synthetic tests, small diagnostics, and output validation.
- Code quality, modularity, documentation, reproducibility, and a technical report
  are part of final judging. Keep the pipeline explainable and fully reproducible.
- Do not privately share competition code or data outside the registered team.
- Run Ruff and the synthetic test suite after code changes.
- Do not commit, push, create a remote, or create a GitHub repository unless the
  user explicitly requests that action.

- **Kaggle PyTorch Performance Invariants**: When generating or modifying Kaggle 
  training scripts for PyTorch, you MUST:
  1. Set `num_workers: 2` (or higher) for DataLoaders on GPU VMs. Never use 0.
  2. Enable `persistent_workers=True` when `num_workers > 0`.
  3. Set `torch.backends.cudnn.benchmark = True` if input dimensions are static.
