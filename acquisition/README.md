# Caption-conditioned visual acquisition: E0--E2

This directory implements the new task in `research/FOLLOWUP_PROPOSAL.md`. It is
separate from the completed phrase-deletion pilot. Synthetic fixtures and model
outputs are engineering evidence only; they are never human ground truth.

## Data boundary

An E0 bundle physically separates `public/` selector inputs from `gold/`
evaluation labels. Selection code receives only `PublicStore.selector_state()`.
Unexecuted crop descriptions, fact IDs, support labels, utilities, and test
answers are not returned. `acquisition.audit` rejects cross-split duplicate
groups, dangling references, public gold-like keys, malformed boxes, cache-key
mismatches, and hand-written action utilities that do not reproduce from atomic
labels.

The staged Flickr30K data is only review material. Flickr entity phrases do not
provide an exhaustive attribute/relation inventory and must not be promoted to
E0 gold without two-person review.

## Reproducible sequence

```sh
# CPU/network preflight: choose images disjoint from the historical 350.
PYTHONPATH=. python -u acquisition/prepare_e0.py \
  --count 100 --seed 3901 --source-split train \
  --output acquisition/data/e0-dev-100

# GPU model-visible baselines; these never read gold.
PYTHONPATH=. python -u acquisition/clip_baseline.py \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/results/clip-e0-dev-100 --device cuda --batch-size 64

# Fixed crop observer, pinned model snapshot and prompt. It resumes by cache key.
PYTHONPATH=. python -u acquisition/observe.py \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/results/qwen-observer-e0-dev-100 \
  --batch-size 8 --max-new-tokens 96

# Stage A: two reviewers independently establish facts/instances.
PYTHONPATH=. python -u acquisition/export_review.py \
  --stage facts \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/data/e0-review-facts

# Adjudicate and freeze canonical entities/facts first. Then stage B maps
# contexts and observer claims to those IDs; it cannot run from source seeds.
PYTHONPATH=. python -u acquisition/export_review.py \
  --stage labels --canonical-bundle acquisition/data/e0-bundle \
  --output acquisition/data/e0-review-labels

# Strong model-visible planner baseline (still not evidence before E0).
PYTHONPATH=. python -u acquisition/planner_baseline.py \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/results/qwen-planner-e0-dev-100 --batch-size 4

# Review-only E2 context drafts; these are never labels or verified captions.
PYTHONPATH=. python -u acquisition/propose_contexts.py \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/results/qwen-context-proposals-e0-dev-100 --batch-size 4

# Blank, independently shuffled review sheets for those drafts.
PYTHONPATH=. python -u acquisition/export_review.py \
  --stage context-proposals --staging acquisition/data/e0-dev-100 \
  --context-proposal-output acquisition/results/qwen-context-proposals-e0-dev-100 \
  --output acquisition/data/e0-review-context-proposals

# Only an adjudicated bundle may pass audit and enter E1/E2.
PYTHONPATH=. python -m acquisition.audit \
  --bundle acquisition/data/e0-bundle --output acquisition/results/e0-audit.json
PYTHONPATH=. python -m acquisition.evaluate \
  --bundle acquisition/data/e0-bundle --baseline-method proposal_score \
  --output acquisition/results/e1-e2.json
```

The two-image fixture exercises leakage and metric invariants:

```sh
PYTHONPATH=. python -m unittest acquisition.test_acquisition -v
```

## Gates

- E0 requires at least 100 images in **each** atomic label family with exactly
  two independent reviews, complete adjudication, and raw pre-adjudication
  agreement at or above 90%. Synthetic fixtures are never eligible.
- E1 uses the recognition oracle over actual frozen-observer outputs and an
  externally frozen, cost-complete strong baseline. Built-in proposal/area/
  center rules are diagnostics and cannot pass the gate. The primary unit is a duplicate-image
  group, not a caption/action row. Oracle minus baseline must be at least 10
  percentage points with a paired interval above zero and no material error-rate
  deterioration.
- E2 reports the caption-necessity gap against the best same-image static action,
  material switches, paraphrase regret, and saturated-caption STOP accuracy.
- E1/E2 failure stops selector training. Oracle is never a deployable method.

All generated data, model caches, raw observations, review packets, and runtime
results are ignored by Git. Compact audited reports can be committed after the
labels are genuinely reviewed.

## Pre-adjudication screening

The `screen_*` commands are deliberately provisional diagnostics. They separate
crop support, text entailment, and complement type, cache every raw model answer,
and report image-clustered intervals. The initial screen shared one Qwen family;
the sensitivity analysis adds pinned SmolVLM visual support and pinned DeBERTa
text novelty, but automated judges still cannot pass E0/E1/E2. The current
diagnostic result and exact limitations are recorded in
`research/PROVISIONAL_DIRECTION_SCREEN.md`.

The required caption-conditioned full-image baseline is reproducible at both
default and low resolution. The low-resolution setting caps Qwen input at 224²
pixels; its outputs are independently screened before semantic typing:

```sh
PYTHONPATH=. python -u acquisition/full_image_completion.py \
  --staging acquisition/data/e0-dev-100 \
  --output acquisition/results/qwen-full-image-completion-lowres-e0-dev-100 \
  --max-pixels 50176 --batch-size 4
PYTHONPATH=. python -u acquisition/screen_full_image_completion.py \
  --staging acquisition/data/e0-dev-100 \
  --completion-output acquisition/results/qwen-full-image-completion-lowres-e0-dev-100 \
  --comparison-output acquisition/results/provisional-independent-both-100 \
  --output acquisition/results/provisional-full-image-completion-lowres-100
PYTHONPATH=. python -u acquisition/screen_completion_core_target.py \
  --staging acquisition/data/e0-dev-100 \
  --full-image-screen-output acquisition/results/provisional-full-image-completion-lowres-100 \
  --comparison-output acquisition/results/provisional-core-target-independent-strict-100 \
  --output acquisition/results/provisional-full-image-completion-lowres-core-strict-100
```

The stopping audit samples 80 stratified context disagreements and emits 160
independently shuffled, blank rows per reviewer. Method names and automated
labels appear only in the private manifest:

```sh
PYTHONPATH=. python acquisition/export_stopping_audit.py \
  --staging acquisition/data/e0-dev-100 \
  --observer-output acquisition/results/qwen-observer-e0-dev-100 \
  --full-image-screen-output acquisition/results/provisional-full-image-completion-lowres-100 \
  --full-image-core-output acquisition/results/provisional-full-image-completion-lowres-core-strict-100 \
  --candidate-core-output acquisition/results/provisional-core-target-independent-strict-100 \
  --output acquisition/data/stopping-audit-full-image-vs-crop-100
```

An adjacent claim-conditioned crop-verification screen is also reproducible:

```sh
PYTHONPATH=. python -u acquisition/screen_claim_crop_support.py \
  --staging acquisition/data/e0-dev-100 \
  --full-image-screen-output acquisition/results/provisional-full-image-completion-lowres-100 \
  --output acquisition/results/provisional-claim-crop-support-lowres-100 \
  --batch-size 8
```

It failed the predeclared continuation rule: proper zoom views retained 80.30%
of supported claims but falsely supported 57.56% of rejected/partial claims.
This cache is a negative direction screen, not training data or evidence for a
crop-verification method.

`export_screen_review.py` converts selected oracle/baseline disagreements into
method-blind crop packets with blank fields for two independent reviewers. It
does not expose model scores or predicted semantic types in reviewer CSV files.

The latest R2 packet adds an image-clustered random population stratum plus
grid judge disagreements, grid/grounded consensus positives, and paired
paraphrase action flips. Public reviewer sheets contain no stratum or automated
label; those fields remain in a separate private manifest:

```sh
PYTHONPATH=. python -m acquisition.export_r2_review \
  --staging acquisition/data/e0-dev-100 \
  --grid-observer acquisition/results/r2-v1-all-contexts-region-conditioned \
  --grid-qwen acquisition/results/r2-v1-strict-core \
  --grid-smol acquisition/results/r2-v1-independent-smol-core \
  --paraphrase-contexts acquisition/results/r2-v3-paraphrases/paraphrased_contexts.jsonl \
  --paraphrase-observer acquisition/results/r2-v3-paraphrase-observations \
  --paraphrase-qwen acquisition/results/r2-v3-paraphrase-core-qwen \
  --grounded-candidates acquisition/results/r1-v2-grounded-candidates.jsonl \
  --grounded-observer acquisition/results/r1-v2-grounded-all-contexts-conditioned \
  --grounded-qwen acquisition/results/r1-v2-grounded-strict-core \
  --grounded-smol acquisition/results/r1-v2-grounded-independent-smol-core \
  --output acquisition/data/e0-r2-blind-review-100

PYTHONPATH=. python -m acquisition.audit_r2_reviews \
  --packet acquisition/data/e0-r2-blind-review-100

# This remains closed until both sheets and all adjudications are complete.
PYTHONPATH=. python -m acquisition.prepare_r3_human_targets \
  --packet acquisition/data/e0-r2-blind-review-100 \
  --output acquisition/results/r3-v2-human-targets
```

The current generated packet has 300 rows per reviewer and is deliberately
`awaiting_reviews`. Blank fields are incomplete work, never negative labels.

R3-v2 label-free feature staging can run while those reviews are pending. It
reuses the frozen R3-v0/v1 caches and encodes only missing context text; it must
not load action outcomes:

```sh
PYTHONPATH=. python -m acquisition.build_r3_feature_store \
  --staging acquisition/data/e0-dev-100 \
  --grid-cache acquisition/results/r3-v0-proxy-siglip/features.pt \
  --grounded-cache acquisition/results/r3-v1-grounded-binding-proxy/features.pt \
  --grounded-candidates acquisition/results/r1-v2-grounded-candidates.jsonl \
  --paraphrase-contexts acquisition/results/r2-v3-paraphrases/paraphrased_contexts.jsonl \
  --completion-output acquisition/results/qwen-full-image-completion-e0-dev-100 \
  --output acquisition/results/r3-v2-feature-store

PYTHONPATH=. python -m acquisition.audit_r3_feature_store \
  --store-output acquisition/results/r3-v2-feature-store \
  --grounded-candidates acquisition/results/r1-v2-grounded-candidates.jsonl
```
