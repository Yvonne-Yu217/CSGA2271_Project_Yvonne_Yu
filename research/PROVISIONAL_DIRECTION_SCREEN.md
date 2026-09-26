# Provisional direction screen before human adjudication

Date: 2026-09-26. This document answers whether the new direction is promising
enough to justify human review. It is **not E0/E1/E2 evidence** and must not be
quoted as a final result.

## Setup and validity boundary

- 100 new Flickr30K images, 500 natural captions, 14 fixed visual observations
  plus STOP per image.
- Frozen Qwen2.5-VL-3B-Instruct observer and two frozen prompted planners: one
  sees the full image plus box coordinates; one sees a numbered montage of the
  actual candidate crops. CLIP inverse similarity and simple rules are included.
- The same Qwen family judges crop support, caption entailment, and semantic
  type. This creates strong self-confirmation risk. In fact, it called all 1,281
  non-empty observation sentences supported and only retained the observer's
  119 `NO_VISIBLE_FACT` outputs as uninformative. Manual Codex spot inspection
  found inference/wording risks and one context-dependent joint-judge error.
- Metrics are binary “contains at least one new correct fact,” not an adjudicated
  count of atomic facts. Bootstrap intervals resample the 100 images after
  averaging their five natural captions.

## E1 screen: any new fact

| Method | Provisional success | 95% image bootstrap | Mean pixel cost |
|---|---:|---:|---:|
| Cost-matched recognition oracle | 99.6% | [99.0, 100.0] | 19,923 |
| Montage prompted planner | 83.6% | [77.8, 88.8] | 63,251 |
| Largest-area/full-image rule | 77.6% | [71.6, 83.2] | 178,573 |
| Cost-matched CLIP inverse | 76.8% | [69.8, 83.2] | 20,809 |
| Cost-matched random expectation | 70.06% | [66.92, 73.07] | 26,014 |
| Coordinate prompted planner | 70.8% | [63.4, 77.8] | 48,326 |

Oracle minus montage is 16.4 points, image-bootstrap interval [11.2, 22.2].
Montage exceeds cost-matched CLIP by 6.8 points [0.6, 13.4] and cost-matched
random by 13.54 points [8.68, 18.12]. Its +6.0 points over the largest-area rule
has interval [−1.0, 13.2], so that comparison is unresolved. The large random
success rate says the unfiltered “any new fact” task is too easy.

A pinned SigLIP-base repeat gave 76.6% for unconstrained inverse similarity and
73.4% when cost-matched. Montage exceeded these by 7.0 points [0.8, 13.6] and
10.2 points [4.0, 16.6], respectively. The semantic-baseline conclusion is not
specific to the original CLIP backbone under this provisional judge.

An independently trained DeBERTa-v3 NLI model was then substituted for Qwen text
entailment (visual support was still the same-family Qwen check). Its binary
new/not-new decisions agreed with Qwen on 73.63% of 7,000 rows and were less
liberal: 81.79% versus 89.24% new. Under DeBERTa, the cost-matched oracle was
99.8%, montage 85.6%, full-image/largest-area 88.2%, and cost-matched CLIP 75.0%.
Oracle minus montage remained 14.2 points [9.8, 19.2]. Montage exceeded
cost-matched CLIP by 10.6 points [3.8, 17.6], but was 2.6 points below full-image
with interval [−8.0, 2.6]. Thus oracle headroom survives an independent text NLI
check; planner superiority over the strongest full-image rule does not.

### Independent visual-support sensitivity

A pinned SmolVLM-Instruct model (`81cd9a775a4d644f2faf4e7becff4559b46b14c7`)
then judged each crop and observation without seeing the full image or caption. It
accepted only 393 of the 1,281 non-empty Qwen observations (30.68%), rejected 888,
and retained 119 deterministic `NO_VISIBLE_FACT` rows. There were no parse
failures. The original same-family judge had accepted all 1,281 non-empty rows,
so its visual-support result was substantially inflated by self-confirmation.

Combining SmolVLM crop support with independent DeBERTa novelty reduced the
cost-matched oracle to 66.2% [57.6, 74.4] and montage to 36.6% [28.6, 45.0].
Cost-matched random was 14.01% [11.59, 16.49] and cost-matched CLIP was 6.6%
[3.2, 10.6]. Oracle minus montage remained 29.6 points [22.2, 37.4], showing
actionable candidate-set headroom. However, the unconstrained largest-area/full-
image rule reached 70.8% [62.8, 78.2], above both montage and the cost-matched
oracle (the latter is constrained to montage's pixel budget). This screen supports
the existence of useful observations, not superiority of the current planner.

## Core target: details of already-mentioned entities

The provisional semantic classifier marked 1,751/7,000 context-actions as
adding an attribute, action, state, or relation of an entity already mentioned;
2,351 primarily introduced a new entity, 1,547 added other scene information,
and 1,351 were not eligible/new.

| Method | Provisional core-target success | 95% image bootstrap |
|---|---:|---:|
| Core-target recognition oracle | 87.8% | [83.0, 92.2] |
| Largest-area/full-image rule | 36.0% | [29.6, 42.6] |
| Montage prompted planner | 30.0% | [24.0, 36.4] |
| Coordinate prompted planner | 27.2% | [20.6, 34.2] |
| Random expectation | 25.01% | [22.24, 27.86] |
| Cost-matched CLIP inverse | 14.8% | [10.2, 19.8] |

Oracle minus montage is 57.8 points [51.2, 64.4]. This is the most encouraging
directional result: the candidate set often contains a potentially useful
mentioned-entity detail, while generic planners rarely select it. It also shows
that the proposed contribution must explicitly optimize this target; generic
“find anything new” selection is not enough.

### Strict-prompt sensitivity check

A second prompt required conservative same-instance linkage rather than a shared
entity category. It substantially reduced all rates:

| Method | Strict core-target success | 95% image bootstrap |
|---|---:|---:|
| Strict core-target oracle | 34.8% | [29.0, 40.8] |
| Montage prompted planner | 7.0% | [4.2, 10.2] |
| Largest-area/full-image rule | 6.6% | [3.8, 9.8] |
| Coordinate prompted planner | 5.0% | [2.4, 8.2] |
| Random expectation | 4.51% | [3.56, 5.60] |
| Cost-matched CLIP inverse | 3.4% | [1.6, 5.6] |

Strict oracle minus montage remains 27.8 points [22.6, 33.2], so the qualitative
headroom survives. But the two prompts agree on only 57.94% of all 7,000 labels;
their mentioned-entity-detail sets have Jaccard 16.65%. The strict set contains
316 rows, 295 of which are also in the broad set, while the broad prompt marks
1,751. Thus the broad estimate is not robust and the strict subset should drive
the first human review. Prompt sensitivity is direct evidence that model-only
semantic labels cannot establish the result.

After applying both independent visual support and independent text novelty, the
strict core-target oracle remained 30.8% [24.4, 37.6], versus montage 7.6%
[4.2, 11.6], largest-area/full-image 10.6% [6.6, 15.2], and random expectation
3.26% [2.47, 4.10]. Oracle minus montage was 23.2 points [17.6, 29.2]. The broad
semantic prompt gave a much higher 73.8% oracle and 22.2% montage. Broad/strict
target-set Jaccard was 25.46%; 222 of 228 strict rows were also broad, but only
25.64% of broad rows survived strict typing. Thus the conservative result still
shows candidate-set headroom, while confirming that the broad formulation is not
a reliable effect-size estimate.

## E2 screen and failure found

Model-authored enriched/paraphrase/saturated captions failed the first E2
screen: caption-necessity gap, natural-to-saturated action switching, and
saturated STOP were all zero. The short “saturated” drafts did not actually
cover the many crop observations. They cannot be used as E2 data without
review/reconstruction.

A deliberately constructed data-sanity condition appended selected observation
text and aggregated supported observation text. It produced a 25-point
caption-necessity gap, 90% selected-action redundancy loss, and 100% STOP/switch
rates under the model judge. Those numbers are partly true by construction and
only show that the evaluator responds when known facts are explicitly changed;
they are not evidence of natural caption dependence.

## Decision

**Continue to targeted human validation, but do not train E3 yet.** The strict
core-target oracle gap survives independent visual-support and text-NLI filters,
so the direction has enough signal to justify review. The current planner is not
a positive method result: it loses to the unconstrained full-image rule on the
independent any-new-fact screen and on strict core-target success. Model-generated
observations, automated semantic typing, binary utility, generic grids, and failed
natural E2 construction still prevent a positive claim.

The highest-value next human task is not a broad 100-image polish pass. First
adjudicate a stratified subset enriched for:

1. core-target oracle choices;
2. montage successes and failures;
3. CLIP/montage disagreements;
4. observer `NO_VISIBLE_FACT`, long, and inferred outputs;
5. generated enriched/paraphrase/saturated contexts.

If human labels preserve a material core-target oracle gap and caption-dependent
action changes, finish E0 on 100 images and freeze a cost-complete strong
baseline. If the gap collapses, revise the observer/fact inventory or stop this
direction before selector training.

A method-blind packet with four selected context-actions per image (400 unique
crop assets, independently shuffled for two reviewers) has been generated in
the ignored data workspace. It prioritizes strict/broad disagreements and
oracle/baseline choices without revealing method identity in the review CSV.
