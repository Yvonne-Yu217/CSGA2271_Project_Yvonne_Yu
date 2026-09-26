# R1–R3 redesign development results

Date: 2026-09-26. These are automated development diagnostics, not human E0
evidence and not a formal confirmation attempt.

## R1 matched observation matrix

The fixed-grid observer was rerun with a shared task instruction under four
conditions: caption-conditioned versus caption-independent, crossed with
region-only versus global-plus-ROI. Each arm contains 1,400 outputs over 100
images and one fixed context per image. Pinned SmolVLM independently screened
visual support; pinned DeBERTa screened novelty; strict Qwen semantic typing was
applied only after both filters.

Caption conditioning materially changed the strict oracle: region-conditioned
reached 89%, versus 70% without the caption, a +19 point paired difference with
95% interval [9, 29]. Global-plus-ROI reached 86%, three points below matched
region-only with interval [-10, 3]. The global view therefore did not repair the
current method. Caption/task alignment, rather than missing global pixels, is
the supported bottleneck hypothesis in this screen.

## R2 same-image caption dependence

The predeclared R2 expansion generated and screened all 7,000 combinations of
100 images, five natural captions per image and 14 fixed-grid actions. Strict
caption-specific recognition-oracle success was 90.4% [86.4, 94.0], versus
80.2% [75.4, 84.8] for the best same-image static action. The paired gain was
10.2 points [7.6, 13.0]. The strong low-resolution full-image completion was
48.6% [41.2, 55.8], leaving a strict oracle gap of 41.8 points [34.8, 49.0].
Across same-image caption pairs, 82.9% changed the successful-action set and the
mean Jaccard was 0.473.

These outcomes pass the automated development continuation rule and support the
claim that available action value depends on the known description. They do
not establish factual correctness: support, novelty and semantic type remain
model judgments, with the semantic classifier sharing the Qwen family with the
observer.

## Strong planners and proxy learnability

The cached caption-conditioned coordinate and montage planners selected a
strictly successful action on only 18.4% and 27.4% of the 500 contexts,
respectively, both below full-image completion. Thus oracle headroom is not
captured by the existing prompted planners.

R3-v0 tested a frozen SigLIP proxy selector on an image-hash 67/14/19
train/validation/proxy-test split. A fixed `lambda=1` elementwise
image–caption-interaction ridge reached 36.84% on the 95 proxy-test contexts,
versus 25.26% image-only and 33.68% full-image completion. Its differences were
+11.58 points [-7.37, 30.53] versus image-only and +3.16 points
[-13.68, 21.05] versus full-image completion. The 94.74% oracle confirms
headroom, but the predictor fails the predeclared uncertainty rule.

Do not tune the ridge coefficient, feature family or split against this proxy
test. The next material change is automatic grounded entity, padded entity and
relation-union views, followed by the same matched observation and residual
evaluation. Human review remains necessary before formal E3 or confirmation.

## Automatic grounded geometry and binding follow-up

Pinned Grounding DINO produced 443 unpadded entity boxes over all 100 images.
The deterministic view builder added 443 15%-padded boxes, 162 nearby-pair
unions, full image and STOP. Grounding required FP32; batch 16 exceeded the
40GB A100, while batch 4 completed 100 images in 20.24 seconds with 9.94 GiB
peak torch allocation. Both failed profiling attempts are retained.

Across all 500 captions, the pooled grounded strict oracle was 87.0%, 3.4
points below the fixed-grid 90.4% and within the predeclared 5-point
noninferiority margin. It exceeded full-image completion by 38.4 points
[31.4, 45.4]. Unpadded entity boxes alone reached 73.0% and exceeded full-image
completion by 24.4 points [16.2, 32.6]. Padding reduced the oracle to 64.4%; the
relation union family reached only 45.4%. These variants are rejected.

The grounded pooled caption-specific oracle gain over its best same-image
static action was 4.8 points [3.0, 6.8]; the entity-only gain was 3.2 points
[1.6, 5.2]. Grounded boxes retain real automated headroom with fewer actions,
but less caption dependence than the grid inventory.

R3-v1 added the detector phrase embedding as an explicit binding feature. On
the already-inspected proxy split, image–caption interaction reached 36.84%,
while phrase–caption and combined interactions each reached 29.47%. The
combined-minus-image-interaction interval was [-21.08, 6.32]. Explicit detector
phrases therefore do not repair selection. Do not tune this reused split.

## Independent semantic-type sensitivity

Pinned SmolVLM independently repeated the final semantic-type decision after
the existing visual-support and novelty gates. On fixed-grid actions it found a
93.0% oracle and an 86.2% best same-image static action, a +6.8 point
caption-specific gain [4.6, 9.2]. On pooled grounded actions it found an 89.4%
oracle and 86.0% static action, a +3.4 point gain [1.8, 5.0]. Binary agreement
with Qwen on eligible claims was 88.6% for grids and 91.2% for grounded views.
The effect sizes vary, but the independent sensitivity does not remove the
caption-dependent action-value result. Neither model is human ground truth.

## Equal-call full-image refinement control

A frozen second full-image call received the original caption and the first
high-resolution completion, then requested one distinct additional fact. All
500 contexts completed. Independent visual and NLI screening, with the NLI
premise containing both prior texts, retained 200/500 distinct supported facts.
Strict mentioned-entity success was 34.6% [28.0, 41.4] under Qwen semantic
typing and 39.8% [32.8, 47.0] under independent SmolVLM typing; their binary
agreement on the 200 eligible outputs was 86.0%.

The first high-resolution full-image call succeeded on 51.0%; the Qwen-typed
union of two full-image calls reached 59.2% [51.6, 66.8]. The fixed-grid oracle
still exceeded that equal-call control by 31.2 points [24.0, 38.4], and the
pooled grounded oracle exceeded it by 27.8 points [20.2, 35.4]. Thus the
automated region headroom is not explained by merely allowing a second
full-image call. This remains a development diagnostic pending human E0.

## Paraphrase stability

SmolVLM deterministically generated one wording change for each of the 500
development captions. The frozen region observer and full independent screening
chain were rerun over all 7,000 context-action pairs. Qwen-typed oracle success
changed from 90.4% to 88.0% (paired change -2.4 points, interval [-4.6, -0.2]);
SmolVLM-typed oracle success changed from 93.0% to 91.6% (-1.4 points,
[-3.8, 0.6]). Oracle availability agreed for 93.6% and 94.6% of contexts,
respectively, but successful-action-set Jaccard was only 0.701 and 0.718.

Independent bidirectional NLI marked 435/500 paraphrases as mutually entailed,
64 as neutral in at least one direction, and one as contradictory in one
direction. On the 435 automatically preserved pairs, Qwen oracle change was
-1.53 points [-4.0, 0.9] and SmolVLM change was -0.15 points [-2.25, 1.95];
action-set Jaccard rose only to 0.718 and 0.736. Thus oracle availability is
largely wording-stable after automated preservation filtering, while individual
successful actions remain moderately unstable. Human preservation review is
still required, and a learned method should treat action utility as set-valued
or noisy rather than a single deterministic target.
