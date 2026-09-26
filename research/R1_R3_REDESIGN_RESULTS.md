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
