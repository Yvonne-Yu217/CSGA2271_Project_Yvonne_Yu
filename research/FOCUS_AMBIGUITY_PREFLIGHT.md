# Focus-ambiguity direction preflight

> Archived side-branch evidence. The complementary-information objective resumes under [METHOD_REDESIGN_PLAN.md](METHOD_REDESIGN_PLAN.md). Historical topic-change recommendations below are superseded; retain the recorded negative results and split caveats. The later paired protocol repurposed official test records, so its internal split must not be presented as untouched official test evaluation.

Date: 2026-09-26. This is a direction screen, not a positive-result claim.
The compact machine-readable evidence snapshot is
`research/focus_ambiguity_preflight_metrics.json`; raw predictions and features
remain in ignored resumable caches.

## Motivation and prior-art boundary

After both complementary-observation selection and claim-conditioned crop
verification failed their predeclared screens, this preflight switches to an
externally labeled CV task rather than relabeling the same proxy. The official
[VQ-FocusAmbiguity task](https://vizwiz.org/tasks-and-datasets/focus-ambiguity-in-visual-questions/)
asks whether a visual question has multiple plausible focus regions and where
all such regions are.

The task itself is not novel. The ICCV 2025 paper
[Acknowledging Focus Ambiguity in Visual Questions](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Acknowledging_Focus_Ambiguity_in_Visual_Questions_ICCV_2025_paper.pdf)
introduced the dataset and benchmarked ambiguity recognition plus plausible
focus localization. The CVPR 2026 workshop paper
[Focus Ambiguity in Visual Questions: A Disambiguation Problem, Not Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2026W/MAR/html/Tseng_Focus_Ambiguity_in_Visual_Questions_A_Disambiguation_Problem_Not_Instance_CVPRW_2026_paper.html)
already replaces strict mask-IoU framing with sufficiency-oriented evaluation
and a two-stage baseline. Consequently, ambiguity detection, segmenting every
focus, or merely adopting point/sufficiency evaluation cannot be this project's
contribution.

A continuation would need a stronger distinction, such as calibrated selective
ambiguity alerts that remain valid under source shift and drive a measurable
clarification benefit. Even that is only a hypothesis: calibration and active
clarification have broad prior art and need a separate literature/falsification
gate before method training.

## Dataset audit

The downloaded official data contain 5,474 records: 70 train, 70 validation,
and 5,334 test. There are 3,227 distinct image filenames and 15,361 mask PNGs.
Every mask is nonempty, binary, and exactly matches the decoded source image
dimensions. Archive CRC checks passed and the audit records SHA-256 hashes for
both archives and all three JSON files.

The audit found source quirks that affect evaluation but not mask validity:

- all 5,474 rows say `set: train`; split must come from the JSON filename;
- 40 image filenames occur in more than one official split, so internal
  cross-validation must group by image;
- no exact normalized image-question pair crosses official splits;
- 11 records have JSON width and height swapped, while image and mask pixels
  agree exactly;
- ambiguity cannot be reconstructed from mask multiplicity. For example, two
  ambiguous rows have one mask, while many unambiguous rows have multiple masks.

The structural audit status is `pass_with_warnings`, with zero hard errors.

## Frozen zero-shot result and label-order falsification

Pinned `HuggingFaceTB/SmolVLM-Instruct` revision
`81cd9a775a4d644f2faf4e7becff4559b46b14c7` classified all 140 public
train+validation rows. An initial ordinary instruction made the multimodal model
answer the embedded VQA question; a format smoke test therefore froze an A/B
classification prompt that explicitly forbids answering it. No output was
manually relabeled.

| Model / input / code order | Accuracy | Balanced accuracy | Ambiguous recall | Image-grouped 95% CI (balanced accuracy) |
|---|---:|---:|---:|---:|
| Always unambiguous | 57.14% | 50.00% | 0.00% | — |
| SmolVLM question, base | 43.57% | 44.58% | 51.67% | [36.32, 52.96]% |
| SmolVLM image + question, base | 58.57% | 51.67% | 3.33% | [50.00, 54.24]% |
| SmolVLM image + question, A/B swapped | 60.71% | 56.04% | 23.33% | [49.63, 62.55]% |
| Qwen2.5-VL-3B image + question, base | 54.29% | 53.96% | 51.67% | [45.55, 62.42]% |
| Qwen2.5-VL-3B image + question, A/B swapped | 57.86% | 51.04% | 3.33% | [48.72, 53.85]% |

With the base prompt, SmolVLM predicted `ambiguous` only twice, both correctly,
and missed 58/60 ambiguous rows. Its 1.67-point balanced-accuracy gain over the
majority rule is not a useful ambiguity recognizer. Image+question minus
question-only was +7.08 points with image-grouped 95% interval
[-1.76, 15.90], so the apparent visual gain is also not reliable.

The A/B order check falsifies these generation-based classifiers more directly.
Swapping which letter denotes ambiguity changed SmolVLM's image predictions
from 2 to 23 ambiguous rows and Qwen's from 66 to 3. Semantic prediction
agreement across code orders was 82.14% for SmolVLM and only 55.00% for Qwen.
Qwen's base text-only run selected `ambiguous` for all 140 rows, while its
swapped run selected that class only four times. These models are following code
and position preferences to a material degree; none of their point scores is an
accepted strong baseline.

To rule out free-generation parsing as the cause, a stronger check read the
next-token A/B logits under both prompts, mapped each margin back to semantic
ambiguity, and averaged the two margins. Qwen then reached only 52.92% balanced
accuracy [47.89, 58.21]% with 13.33% ambiguous recall. SmolVLM reached 50.83%
[50.00, 52.73]% and detected 1/60 ambiguous rows. Even at the logit level, base
versus swapped semantic predictions agreed on only 57.14% of image-question
rows for Qwen and 24.29% for SmolVLM. The failure is not a decoding artifact.

## Fixed SigLIP learnability probe (train to validation)

Pinned `google/siglip-base-patch16-224` revision
`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` supplied frozen features. One
fixed dual ridge classifier (`lambda=1`, no tuning) was fit on 70 train rows and
evaluated on 70 validation rows.

| Frozen features | Validation accuracy | Balanced accuracy | Ambiguous recall | 95% bootstrap CI (balanced accuracy) |
|---|---:|---:|---:|---:|
| Question | 52.86% | 52.50% | 50.00% | [40.83, 64.29]% |
| Image | 51.43% | 52.50% | 60.00% | [40.83, 64.29]% |
| Image + question | **62.86%** | **60.83%** | 46.67% | [49.48, 72.02]% |

The joint probe is 10.83 balanced-accuracy points above the majority rule, but
its interval crosses 50%. Results are strongly source-dependent: joint-probe
accuracy is 80% on VizWiz, 45% on COCO, and 50% on PACO; all ten MSRA validation
rows are unambiguous, on which it reaches 90%. The aggregate improvement could
therefore reflect small-sample/source structure rather than general focus
reasoning.

A pinned CLIP-base replication makes that confound concrete. Its fixed
image-only probe reached 62.50% balanced accuracy with interval
[51.19, 73.56]%, whereas image+question reached only 57.50% with interval
[45.67, 69.05]%. Question-only reached 59.17% [48.21, 70.24]%. The fact that
discarding the question gives the strongest CLIP result contradicts an
interpretation based on question-specific focus reasoning. Source breakdown is
also diagnostic: image-only accuracy is 40% on COCO, 50% on PACO, 75% on
VizWiz, and 100% on the ten all-unambiguous MSRA rows. The apparent positive
aggregate can be driven by source/domain recognition.

SigLIP feature extraction took 7.16 seconds on one L4 and peaked at 474.90 MiB
of torch GPU allocation; the CLIP replication took 10.54 seconds and peaked at
384.14 MiB. Base Qwen and swapped Qwen each peaked at 10.79 GiB and took
about 96.6 seconds end-to-end; base and swapped SmolVLM peaked at 4.95 GiB and
took 55.9 and 54.8 seconds. The symmetric-logit runs took 177.69 seconds /
16.14 GiB for Qwen and 92.03
seconds / 5.03 GiB for SmolVLM on allocation job 1998. All artifacts are cached
under ignored `focus_ambiguity/results/`. The official test set remains locked:
no test inference, threshold selection, or performance inspection was done.

## Decision

This direction is **rejected in its current form**, not merely awaiting a larger
training run. Prompted generative classification is invalidated as a strong
baseline by label-order sensitivity.
There is some train-to-validation signal in the frozen probes, but the sample is
too small, the multimodal interval includes chance, source behavior is
inconsistent, and the CLIP image-only shortcut is stronger than its multimodal
probe. Recent papers already cover the base task and sufficiency-oriented
localization.

Do not submit test predictions, scale training, or use this as the new main
topic. The base task and sufficiency framing are already covered; zero-shot
generation and symmetric logits fail; the only learned signal is small and
confounded by an image-only/source shortcut. Archive the reproducible screen and
move to another externally verifiable CV task. Reopening focus ambiguity would
require a genuinely new contribution and new source-balanced development data,
not tuning these 140 rows.
