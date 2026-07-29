# Competition reference

Verified on 2026-07-29 from the authenticated Kaggle competition pages and
read-only Kaggle CLI.

Official competition:
<https://www.kaggle.com/competitions/filament-segmentation-2026>

## Task

Generate an individual segmentation mask for every detected solar filament in
full-disk GONG H-alpha observations. Filaments are dark, fine-scale, often
fragmented structures whose morphology and continuity matter.

The supplied MAGFiLO data is COCO-style. It includes polygons, bounding boxes,
spines, and chirality categories, but the competition target is segmentation.

## Data inventory

The authenticated API lists 888 files and approximately 716.18 MiB of content:

| Item | Count | Listed size |
| --- | ---: | ---: |
| Training JPEGs | 707 | 533.46 MiB |
| Test JPEGs | 180 | 136.36 MiB |
| Training annotation JSON | 1 | 46.36 MiB |

The downloaded ZIP is 670.98 MiB; the API-listed extracted content totals
716.18 MiB.

All observations are 2048 × 2048, grayscale, 8-bit JPEG conversions of original
FITS observations. File stems encode `YYYYMMDDHHMMSSII`, where the final
instrument code identifies the observatory.

The annotation JSON has the COCO collections `info`, `images`, `annotations`,
`licenses`, and `categories`. Each annotation includes:

- string `id`;
- `image_id`;
- `category_id`;
- one closed polygon in `segmentation`;
- polygon `area`;
- filament `spine`;
- `[x, y, width, height]` bounding box;
- `iscrowd`, always zero.

Categories are Left (1), Right (2), Unidentifiable (3), and Ambiguous (4).
Those chirality labels are not the competition prediction target.

One physical JPEG may be annotated independently by multiple annotators. Its
COCO `image["id"]` combines an annotator batch with the observation stem, while
`image["file_name"]` identifies the pixels. Group validation by `file_name` to
prevent identical observations from crossing folds.

The full local structural audit found:

- 1,154 COCO image records for 707 physical training observations;
- 411 observations with one annotation set, 145 with two, and 151 with three;
- 447 additional annotation-set records caused by repeat annotators;
- 8,199 filament annotations: 2,535 Left, 2,590 Right, and 3,074
  Unidentifiable; the defined Ambiguous category has zero annotations;
- unique image-record and annotation IDs, no orphan annotations, valid 2048 ×
  2048 metadata, one valid polygon per annotation, positive areas and boxes, and
  `iscrowd=0` throughout;
- no shared observation stems or exact SHA-256 image hashes between train and
  test, and no byte-identical duplicate files within either split.

The official prose describes polygons as explicitly closed. The actual float
coordinates do not always repeat the first point exactly; COCO polygon
rasterization closes them implicitly. Validation therefore checks polygon
structure, finite in-bounds coordinates, and pycocotools conversion instead of
requiring exact endpoint equality.

## Evaluation

Final judging uses:

- 70% quantitative comparison:
  - mean Dice via `torchmetrics.segmentation.DiceScore`;
  - Dice distribution;
  - IoU distribution;
  - one-to-many and many-to-one relations between ground truth and predictions;
- 30% qualitative comparison:
  - complete pipeline description;
  - apparent morphology of predicted masks on H-alpha images;
  - code modularity and documentation.

The public scoreboard shows mean Dice for approximately 50% of test images.
The remaining image results are visible only to the organizers. Predicted and
ground-truth instances are matched by actual overlap, not instance suffix.

The organizer has not published a complete local implementation of all matching
and qualitative judging in the page text. Local metrics must therefore be
reported as diagnostics unless exact parity is demonstrated.

## Submission format

Upload one CSV with exactly:

```text
filament_id,segmentation_rle
20150125172714Mh_1,<COCO compressed RLE counts>
20150125172714Mh_2,<COCO compressed RLE counts>
```

The base of `filament_id` must be the test image ID. A unique suffix such as
`_1`, `_2`, and `_3` identifies predicted instances. The suffix does not need to
match ground-truth ordering.

`segmentation_rle` contains only the ASCII COCO compressed `counts` string. Do
not serialize the RLE `size` field and do not embed quote characters. Every mask
is fixed at 2048 × 2048. Use `pycocotools` and Fortran-contiguous binary masks.

## Competition-specific rules

- Sponsor: U.S. National Science Foundation National Solar Observatory.
- Prize pool: up to USD 3,000, dynamically divided among the top three.
- Maximum team size: five.
- Maximum submissions: five per day.
- Final submissions selectable: two.
- One Kaggle account and one competition team per participant.
- Team mergers must remain within the five-person limit, must occur before the
  applicable merger/competition deadline, and the merged teams' combined
  submission count may not exceed the allowance for one team at that time.
- No private sharing of competition data or code outside the registered team.
- Public competition-code sharing during the competition must be made available
  to all participants through the competition's Kaggle forum or notebooks and
  carries an OSI-approved license.
- Competition data is for non-commercial competition use, academic research, and
  education only.
- External data and models are generally permitted only when public, equally
  accessible, and free or reasonably accessible at minimal cost.
- The evaluation page additionally says models may not use any other
  ground-truth metadata for training and may use only the provided H-alpha test
  images at inference. Conservatively, this repository disables external labeled
  data and supervised pretrained weights until the host clarifies the wording.
- Automated ML and AI tools are allowed, but a prize-eligible participant must
  understand, explain, justify, and reproduce every major component.
- Winners must deliver reproducible training/inference code, documentation, the
  compute requirements, and a detailed technical report.
- A winning submission and its code must use the open-source licensing required
  by the competition-specific winner terms.
- Participants supervised, advised, or directly mentored by an organizer may
  participate but are not prize-eligible.
- Prize recipients must complete sponsor banking/vendor onboarding and required
  tax/acceptance documents; payment depends on the ability to receive a U.S.
  bank transfer.
- The competition-specific governing-law provision selects California law and
  courts in Santa Clara County.

This is an engineering summary, not a replacement for the binding
[official rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules).
See [external-data.md](external-data.md) for the repository's conservative
interpretation of the label restriction.

## Open-access requirement

Final-evaluation participants must maintain a source-code Git repository, make
it public immediately at competition close, and keep it public at least until
the winners are announced. Code quality and reproducibility affect judging.

No repository license is selected yet because that is a user/legal choice. A
compatible OSI-approved license must be selected before any required public
release or prize delivery.

## Timeline

- 2026-07-10: competition launch.
- 2026-11-15 at 06:00 UTC (11:30 IST): Kaggle API deadline for final reports
  and solutions.
- 2026-11-30: winners announced.
- 2026-12-14 through 2026-12-17: optional IEEE BigData conference in Phoenix.

Always confirm the live Kaggle timeline before a deadline-sensitive action.

## Primary references

- [Kaggle overview and evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026)
- [Kaggle data page](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
- [Kaggle rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules)
- [MAGFiLO Scientific Data paper](https://doi.org/10.1038/s41597-024-03876-y)
- [COCO format](https://cocodataset.org/#format-data)
- [pycocotools](https://pypi.org/project/pycocotools/)
- [TorchMetrics DiceScore](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html)
