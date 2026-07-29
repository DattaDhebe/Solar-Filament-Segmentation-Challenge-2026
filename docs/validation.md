# Validation policy

## Split unit

Use the physical observation `file_name` as the group key. Do not split on the
COCO annotation-set `image["id"]`; separate annotators can produce different IDs
for the same JPEG, which would leak identical pixels between folds.

Keep one immutable fold assignment for comparable experiments. Record the seed,
fold count, grouping key, row ordering, and a fingerprint of the assignments.

## Selection boundary

Fit every learned transform on the training portion of each fold. This includes
normalization statistics, image-quality filtering, sampling weights, calibration,
mask thresholds, component-size cutoffs, morphology parameters, and instance
separation.

Test images may be loaded only by final inference code after all choices are
frozen. Do not tune against public-leaderboard feedback.

## Diagnostics

At minimum, store per-fold and aggregate:

- mean and distribution of per-instance Dice;
- mean and distribution of IoU;
- per-image number of ground-truth and predicted instances;
- unmatched ground truth and predictions;
- one-to-many and many-to-one overlap counts;
- performance by instance area and observatory code;
- inference runtime and peak memory;
- visual panels sampled from validation without cherry-picking.

The exact organizer matching code is not available in the competition page text.
Document the local matching algorithm and thresholds explicitly and do not call a
diagnostic an exact leaderboard replica without proof.

## Candidate promotion

A candidate should be promoted only when it improves the fixed grouped OOF
evaluation, has stable folds and subgroups, does not materially worsen missed or
fragmented filaments, and produces visually plausible fine structures. One
public-score increase is not sufficient evidence.
