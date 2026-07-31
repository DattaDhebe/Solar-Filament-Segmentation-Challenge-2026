# Experiment registry

Generated metrics and predictions belong under ignored `outputs/`; this file
stores only concise, reviewable decisions.

| ID | Change | Validation | Result | Decision |
| --- | --- | --- | --- | --- |
| 000 | Workspace, data integrity, schema, RLE, and submission smoke tests | Full read-only data audit plus synthetic tests | 23 tests passed; 707/180 images verified; no exact split overlap | Infrastructure accepted |
| 001 | 512 × 512 small U-Net trained from scratch; semantic foreground separated into connected-component instances | Fixed fold 0 of five, grouped by physical `file_name`; fold fingerprint `69a31d113fd4e4cea63f1a072d94dd0dea2c1432c18d349d8a2575c5d1915aa7` | Best semantic Dice 0.6403; 1,970 canonical RLE instances; submission `55089873` scored 0.52 | Retain only as the first end-to-end baseline; instance-matching OOF diagnostics are still required |
| 002 | Five 1024 × 1024 small U-Nets trained from scratch and ensembled; 27 OOF post-processing candidates | Five fixed grouped folds with the same fingerprint as 001 | Mean best-fold semantic Dice 0.6481; penalized OOF instance Dice 0.4304; matched Dice 0.7047; matched IoU 0.5635; 1,033 missed and 4,432 extra; submission `55099193` scored 0.62 | Promote over 001, but retain as a baseline because over-segmentation remains substantial |
| 003 | Continue all five 002 checkpoints with soft annotator-consensus targets, four-way TTA, and 72 OOF instance-filter candidates | Same five immutable grouped folds; external labels and supervised pretrained weights disabled | TTA semantic Dice 0.6527; penalized OOF instance Dice 0.4901; matched Dice 0.7215; matched IoU 0.5819; 1,693 missed and 2,103 extra; submission `55107032` scored 0.66 | Promote over 002; retain the missed/extra tradeoff as the main next optimization target |
| 004 | Reuse frozen 003 checkpoints; cross-fit an OOF component-quality regressor and screen conservative fragment linking | Same five immutable grouped folds; every ranker score is cross-fitted by validation fold; selection uses OOF annotations only | Kaggle run was externally cancelled after OOF and during final test encoding; OOF Dice 0.5015, 1,807 missed, 1,673 extra; no complete candidate | Retain the evidence, but do not submit or rerun unchanged because the recall gate failed |
| 005 | Fine-tune frozen-source 003 checkpoints with hybrid soft-consensus and deterministic annotator-specific targets; screen two fixed loss pairs on folds 0-1 and reuse the selected models for five-fold confirmation | Same immutable grouped folds and fingerprint; fixed experiment-003 post-processing isolates the supervision change | Completed in 98.5 minutes; OOF Dice 0.4905, matched Dice 0.7225, matched IoU 0.5830, 1,609 missed and 2,309 extra; 1,386 valid test masks | Do not promote: recall improved, but Dice stayed below 0.500 and extras exceeded the 003 baseline |
| 006 | Blend experiment-003 and experiment-005 probabilities, then select stricter component filtering from a 135-candidate OOF-only grid | Same immutable grouped folds; both source checkpoints must reproduce their recorded OOF references before selection; test remains inference-only | Prepared locally; Kaggle result and leaderboard score are not yet available | Promote only if OOF Dice reaches 0.495, misses are at most 1,692, and extras are at most 2,103 |

For each real experiment, record:

- source commit and YAML config;
- data/fold fingerprint;
- one intentional model or pipeline change;
- grouped OOF Dice/IoU and fold spread;
- fragmentation and missed/extra instance diagnostics;
- runtime and actual device;
- decision and honest rationale.
