# Generated outputs

Create one directory per experiment:

```text
outputs/experiments/<experiment-id>-<UTC-timestamp>/
```

Each real run should retain, outside Git:

- a copy of the resolved YAML configuration;
- environment and source revision metadata;
- fixed validation fold assignments;
- fold and aggregate Dice/IoU diagnostics;
- instance matching, missed/extra instance, and fragmentation diagnostics;
- thresholds and post-processing parameters selected from validation only;
- model checkpoints and OOF/test predictions;
- inference runtime and hardware details;
- the generated submission and its validation report.

Only this README is tracked. Everything else below `outputs/` is ignored.
