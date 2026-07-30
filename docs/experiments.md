# Experiment registry

Generated metrics and predictions belong under ignored `outputs/`; this file
stores only concise, reviewable decisions.

| ID | Change | Validation | Result | Decision |
| --- | --- | --- | --- | --- |
| 000 | Workspace, data integrity, schema, RLE, and submission smoke tests | Full read-only data audit plus synthetic tests | 23 tests passed; 707/180 images verified; no exact split overlap | Infrastructure accepted |
| 001 | 512 × 512 small U-Net trained from scratch; semantic foreground separated into connected-component instances | Fixed fold 0 of five, grouped by physical `file_name`; fold fingerprint `69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7` | Best semantic Dice 0.6403; 1,970 canonical RLE instances; submission `55089873` scored 0.52 | Retain only as the first end-to-end baseline; instance-matching OOF diagnostics are still required |
| 002 | Five 1024 × 1024 small U-Nets trained from scratch and ensembled; 27 OOF post-processing candidates | Five fixed grouped folds with the same fingerprint as 001 | Mean best-fold semantic Dice 0.6481; penalized OOF instance Dice 0.4304; matched Dice 0.7047; matched IoU 0.5635; 1,033 missed and 4,432 extra; submission `55099193` scored 0.62 | Promote over 001, but retain as a baseline because over-segmentation remains substantial |
| 003 | Continue all five 002 checkpoints with soft annotator-consensus targets, four-way TTA, and 72 OOF instance-filter candidates | Same five immutable grouped folds; external labels and supervised pretrained weights disabled | Private Kaggle run active; no result or score claimed | Pending grouped OOF diagnostics and artifact review |

For each real experiment, record:

- source commit and YAML config;
- data/fold fingerprint;
- one intentional model or pipeline change;
- grouped OOF Dice/IoU and fold spread;
- fragmentation and missed/extra instance diagnostics;
- runtime and actual device;
- decision and honest rationale.
