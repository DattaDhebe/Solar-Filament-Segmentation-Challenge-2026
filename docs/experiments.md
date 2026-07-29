# Experiment registry

Generated metrics and predictions belong under ignored `outputs/`; this file
stores only concise, reviewable decisions.

| ID | Change | Validation | Result | Decision |
| --- | --- | --- | --- | --- |
| 000 | Workspace, data integrity, schema, RLE, and submission smoke tests | Full read-only data audit plus synthetic tests | 23 tests passed; 707/180 images verified; no exact split overlap | Infrastructure accepted |

For each real experiment, record:

- source commit and YAML config;
- data/fold fingerprint;
- one intentional model or pipeline change;
- grouped OOF Dice/IoU and fold spread;
- fragmentation and missed/extra instance diagnostics;
- runtime and actual device;
- decision and honest rationale.
