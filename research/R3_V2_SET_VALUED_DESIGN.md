# R3-v2: set-valued caption-conditioned action selection

Date: 2026-09-26. Status: **prepared behind the human E0 gate; not trained**.

## Why the method family changes

R3-v0's pointwise frozen-SigLIP ridge did not clear its proxy uncertainty rule.
R3-v1 added explicit detector phrase binding but was worse on the same inspected
split. Further ridge or feature-coefficient tuning is prohibited. R2-v3 also
shows why a single hard best-action label is poorly specified: oracle
availability agrees under automatically preserved paraphrases, but the positive
action sets have only about 0.72--0.74 Jaccard.

The replacement is a joint set scorer, not another independent candidate
regressor. MDETR motivates text-modulated region reasoning rather than a frozen
detector used as an unrelated black box ([paper](https://arxiv.org/abs/2104.12763),
[official code](https://github.com/ashkamath/mdetr/tree/ea09acc44ca067072c4b143b726447ee7ff66f5f)).
BLIP's retrieval objective distributes target mass over multiple positives and
uses soft targets ([paper](https://arxiv.org/abs/2201.12086),
[official implementation](https://github.com/salesforce/BLIP/blob/056a169437371659074aa2732649d5de3bffb4a8/models/blip_retrieval.py)).
RankFormer supplies an explicit listwise/listwide ranking precedent
([paper](https://arxiv.org/abs/2306.05808),
[author implementation](https://github.com/maartenbuyl/rankformer/tree/f2722352575079ec37b18e9a1f2fac170cda585e)).
These sources motivate components; none is claimed to solve complementary fact
selection directly.

## Frozen design

- Unit of prediction: every candidate list for one image-caption state,
  including full image and STOP.
- Candidate token: frozen crop visual embedding, frozen full-image embedding,
  geometry, action-family indicator, detector phrase embedding where available,
  and measured visual cost. Context token: frozen caption embedding plus the
  first full-image completion state when R2 evaluation requires it.
- Model: two-layer permutation-equivariant Transformer encoder, 256 hidden
  units and four heads, followed by one score per action. No candidate position
  embedding is allowed.
- Target: uniform probability over all independently human-valid actions.
  Unknown actions are masked, not made negative. STOP is positive only when no
  reviewed non-STOP action is valid.
- Loss: multi-positive listwise cross entropy plus symmetric KL consistency for
  caption/paraphrase pairs that humans mark semantically equivalent. The
  consistency coefficient is fixed from training folds only; the initial value
  is 0.25 and it may be changed once using nested development folds before the
  model is frozen.
- Splitting: image-grouped five-fold development only, three fixed seeds. The
  old 67/14/19 proxy split is not used for tuning or selection. Formal
  confirmation requires a fresh image-disjoint set and consumes the registered
  confirmation budget.

## Comparators and ablations

Required comparators are full-image completion, equal-call full-image
refinement, frozen prompted coordinate/montage planners, the recorded R3-v0
ridge, and a pointwise MLP with the same inputs and parameter budget. Ablations
remove list interaction, multi-positive targets, paraphrase consistency,
caption features, detector phrase features, and STOP. Report inference calls,
visual tokens, latency, peak memory, and selected-action success.

## Gate and decision

Training cannot start until the new R2 packet has two independent reviews and
adjudication, and label construction passes leakage/schema audit. Automated
Qwen/Smol labels may be used only for sensitivity, never as the primary target.
On cross-validated human development labels, continue only if the set model
beats the strongest same-input learned comparator by at least 3 percentage
points and the image-cluster paired 95% interval excludes zero, while retaining
paraphrase noninferiority within 3 points. Otherwise archive R3-v2 and move to
the global fact-selection family rather than tune this selector indefinitely.

The implemented module contains only the architecture and preregistered losses.
It performs no data loading or training until the E0 gate opens.

## Label-free feature staging

The frozen feature store is prepared at
`acquisition/results/r3-v2-feature-store/feature_store.pt` with SHA-256
`27d08f2a8ee24dccbc5339f94ecc212841caf0365bb3824001981b8ef651b09a`.
It covers 100 images, 500 original/paraphrased/full-image-completion context
states, and 643 grounded-entity/full-image/STOP actions. Candidate vectors have
2,315 dimensions: frozen crop, full-image and detector-phrase SigLIP blocks,
geometry, action family, proposal score and normalized cost. A separate audit
verified finite/unit-normalized blocks, unique IDs, valid per-image offsets,
family semantics, cost bounds and absence of outcome-label inputs. This cache
prepares computation only; it is not evidence and does not open the human gate.
