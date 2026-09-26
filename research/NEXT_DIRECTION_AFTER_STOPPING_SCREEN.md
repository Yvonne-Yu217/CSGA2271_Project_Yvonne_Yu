# Next-direction screen after the full-image baseline

> Archived method screen. The current plan is [METHOD_REDESIGN_PLAN.md](METHOD_REDESIGN_PLAN.md). Its rejection remains recorded, but the instruction below to change CV topic is superseded. Here, “false support” is measured against full-image model judgments, not adjudicated human truth.

Date: 2026-09-26. This is a falsification plan, not a new positive-result claim.

## Why the current action design stops

The current fixed crop inventory cannot beat caption-conditioned low-resolution
full-image completion on the primary strict mentioned-entity-detail proxy, even
with oracle crop selection. Training a selector on those actions is therefore
gated off. The remaining any-new-fact oracle margin is not enough to rescue the
primary claim.

## Candidate pivot: selective evidence verification

A defensible adjacent question is whether a low-resolution VLM should trust a
candidate fact or acquire one high-resolution view to verify it. The state is
`(low-resolution image, existing caption, proposed atomic claim)`; the action is
a crop or STOP; the outcome is supported, contradicted, or insufficient evidence.
The useful target is not generic hallucination detection. It is evidence for an
attribute or relation of the **same mentioned entity instance**, especially when
multiple instances share a category.

This pivot is motivated by the observed failure rather than chosen after seeing
a favorable score: the 50,176-pixel full-image generator produced useful claims,
but the independent visual screen rejected about 41% of them. A high-resolution
view could be useful only if the candidate inventory contains discriminative
evidence and a deployable selector can find it more efficiently than simply
raising the full-image resolution.

## Novelty boundary from primary sources

Generic coarse-to-fine acquisition is already covered by
[AdaptVision](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_AdaptVision_Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition_CVPR_2026_paper.html),
which lets a VLM answer from a low-resolution image or invoke a crop tool.
[TEVA](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.html)
and [CropVLM](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Carvalho_CropVLM_Learning_to_Zoom_for_Fine-Grained_Vision-Language_Perception_CVPRW_2026_paper.html)
already learn or construct efficient high-resolution views. More directly,
[Liao et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_Can_Large_Vision-Language_Models_Correct_Semantic_Grounding_Errors_By_Themselves_CVPR_2025_paper.pdf)
show that ROI-crop binary verification improves feedback for semantic grounding
errors. [EVICT](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html)
tests evidence sufficiency by counterfactual region dropout.

Automated judge disagreement is also not itself a novel topic. Recent work
explicitly studies perceptual judgment bias and same-family preference, including
[Philautia-Eval](https://arxiv.org/abs/2604.11589) and
[perceptual perturbation for multimodal judges](https://arxiv.org/abs/2606.02578).
The earlier 100% Qwen self-support versus 30.68% cross-family support is therefore
a validity warning, not a standalone contribution claim.

Consequently, this project must not claim novelty for crop acquisition, binary
verification, or cross-family judging alone. A possible distinction would require
all of the following:

1. claims conditioned on what an existing description already says;
2. explicit same-instance attribute/relation binding with hard same-category
   distractors;
3. evidence-sufficiency/abstention rather than forced yes/no verification;
4. fixed generation and verifier models, with gains attributable to view choice;
5. human-confirmed labels and transfer across generator/verifier families.

## Predeclared screen and stop rules

The current 100-image development set is used only to decide whether to build a
fresh task. For every low-resolution full-image claim, the frozen SmolVLM judge
checks all 14 existing views. The decision uses the 13 proper zoom views that
exclude the full-image candidate; a separate nine-tile result tests strictly
local evidence. Before inspecting the result, continue only if:

- at least 80% of full-image-supported claims have a supporting crop;
- no more than 20% of full-image-rejected claims receive any supporting crop;
- hard same-instance cases retain material coverage rather than only objects;
- a claim-conditioned selector has at least a 10-point oracle gap at lower cost;
- a literature distinction stronger than generic zoom/verification remains.

The first two thresholds are deliberately demanding because an `any crop`
decision incurs 14 opportunities for false support. They do not turn SmolVLM
into ground truth; any positive screen still requires a method-blind human subset.
If crop recall is low or false support is high, discard this pivot without
training. If it passes only on generic objects, it also fails the intended
same-instance contribution.

## If the screen passes

Construct new, image-disjoint data around multiple same-category instances and
atomic attribute/relation claims. Compare low/high-resolution full image,
AdaptVision-style crop calls, ROI binary verification, all-crop verification,
random/area/text-region retrieval, and a claim-conditioned selector. Evaluate
support, contradiction, insufficiency, wrong-instance binding, abstention,
pixels/tokens, and latency. Do not reuse the present automated labels as test
gold.

## If the screen fails

Archive the observation-selection and verification results as negative direction
screens and choose a genuinely different CV task. Do not relabel the same crops,
swap reward weights, or move directly to finance to manufacture a positive
result. Finance remains downstream of a separately validated CV contribution.

## Executed result and decision

The full 500-context/7,000-view screen completed on one L4. Five initial outputs
ignored the label instruction and transcribed `Dunlop`; a deterministic stricter
format retry resolved four, and an A/B/C/D retry resolved the last. All retry
variants and raw outputs remain cached. No label was assigned manually.

The proper-zoom set excluding the full image reached 80.30% recall for claims
that the same frozen SmolVLM supported on the full image, with image-clustered
95% interval [72.68, 87.40]. It simultaneously marked at least one zoom crop as
supporting 57.56% of full-image-rejected/partial claims [47.72, 66.90]. This
grossly exceeds the predeclared 20% false-support ceiling. Restricting evidence
to the nine local tiles reduced false support to 14.38% [8.10, 21.46], but recall
collapsed to 42.76% [33.74, 51.93]. Including the high-resolution full-image
candidate gave 99.70% recall and 62.95% false support, confirming that repeated
views do not solve the reliability problem.

**Decision: this pivot fails its predeclared screen.** Do not train a crop
selector, tune thresholds on these 500 contexts, or claim a verification result.
The same-judge reference is an additional limitation, but cannot rescue a result
whose measured false-support rate misses the stop threshold by almost 38 points.
The next CV task must use externally verifiable labels and a target different
from generic zoom/verification.
