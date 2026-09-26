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
