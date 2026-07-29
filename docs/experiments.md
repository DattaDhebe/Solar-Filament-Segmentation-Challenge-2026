# Experiment registry

Generated metrics and predictions belong under ignored `outputs/`; this file
stores only concise, reviewable decisions.

| ID | Change | Validation | Result | Decision |
| --- | --- | --- | --- | --- |
| 000 | Workspace, data integrity, schema, RLE, and submission smoke tests | Full read-only data audit plus synthetic tests | 23 tests passed; 707/180 images verified; no exact split overlap | Infrastructure accepted |
| 001 | 512 × 512 small U-Net trained from scratch; semantic foreground separated into connected-component instances | Fixed fold 0 of five, grouped by physical `file_name`; fold fingerprint `69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7` | Best semantic Dice 0.6403; 1,970 canonical RLE instances; Kaggle submission `55089873` pending | Retain only as the first end-to-end baseline; instance-matching OOF diagnostics are still required |

For each real experiment, record:

- source commit and YAML config;
- data/fold fingerprint;
- one intentional model or pipeline change;
- grouped OOF Dice/IoU and fold spread;
- fragmentation and missed/extra instance diagnostics;
- runtime and actual device;
- decision and honest rationale.
