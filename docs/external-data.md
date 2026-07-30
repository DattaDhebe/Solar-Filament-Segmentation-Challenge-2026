# External data and pretrained models

## Current policy

External labeled data and supervised pretrained weights are disabled.

The competition-specific rules generally allow public, equally accessible,
free or minimally costly external data and models. The evaluation page is more
specific: inference may use only the supplied test H-alpha images, and models
may not use other ground-truth metadata for training.

On 2026-07-27, organizer Azim Ahmadzadeh clarified that external modalities such
as GONG magnetograms or SDO AIA images may provide segmentation information, but
MAGFiLO ground truth must be limited to the annotations supplied through this
competition; otherwise a submission will be disregarded. The clarification is
in [discussion 729396](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396).
This repository therefore keeps the conservative configuration:

```yaml
model:
  pretrained_weights: false
  external_labeled_data: false
```

## MAGFiLO leakage warning

Do not download the full public MAGFiLO annotations for training. Participants
verified that all 180 test filenames have annotations in the public MAGFiLO
release. Matching those annotations to test filenames would reveal every hidden
target, violate the organizer clarification, and make the submission
ineligible, regardless of the generic external-data allowance.

The official competition download is the only approved label source in this
workspace.

## Public metric warning

The public leaderboard is not a reliable model-selection signal. Participants
reported that all-empty masks can score above 0.9 and even 1.0; see
[discussion 728474](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728474).
The organizer states that leaderboard scores are only a preliminary filter and
that segmentation quality and the full evaluation rubric determine awards.
Never remove valid predictions merely to exploit this scoring bug.

## Approval checklist

Before any future external asset is used:

1. Link a public organizer clarification that permits the exact use.
2. Record its stable URL, license, version, hash, and acquisition date.
3. Confirm it is equally accessible to every participant at no or minimal cost.
4. Audit filenames, timestamps, exact hashes, and perceptual similarity against
   all competition observations.
5. Prove it contains no hidden test annotations or metadata-derived targets.
6. Record how it affects winner open-source and reproducibility obligations.
7. Add one versioned config flag and evaluate it only with fixed grouped OOF
   validation.

Store any approved external files under ignored `data/external/`. Never commit
the files themselves.
