# External data and pretrained models

## Current policy

External labeled data and supervised pretrained weights are disabled for the
initial pipeline.

The competition-specific rules generally allow public, equally accessible,
free or minimally costly external data and models. The evaluation page is more
specific: inference may use only the supplied test H-alpha images, and models
may not use other ground-truth metadata for training. Until the organizer
clarifies how that sentence applies to pretrained weights and unrelated labeled
datasets, the conservative configuration is:

```yaml
model:
  pretrained_weights: false
  external_labeled_data: false
```

## MAGFiLO leakage warning

Do not download the full public MAGFiLO annotations for training. The public
source contains observations beyond this competition's labeled split and may
include the 180 hidden-label test observations. Matching public annotations to
test filenames would reveal targets and violate the train/test boundary,
regardless of the generic external-data allowance.

The official competition download is the only approved label source in this
workspace.

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
