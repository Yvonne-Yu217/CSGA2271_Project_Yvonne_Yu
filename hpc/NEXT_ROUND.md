# Frozen-checkpoint follow-up

This bounded experiment evaluates the existing deletion-trained scorer on deletion, matched generalization, and natural captions without retraining. It retains three individual seeds, their metric mean, and a true prediction ensemble as distinct results. It does not establish that natural-caption changes preserve other entities.

## Prerequisites and budget

Use the existing HPC dataset, model cache, and deletion run with all three `rank_local_control_seed*.pt` checkpoints, `manifest_snapshot.json`, `feature_runtime.json`, and `metrics.json`. Raw checkpoints are intentionally not in Git. Imported documents record a 24 GPU-hour cap, while the earlier local authorization was 20 GPU hours. Follow `GOAL.md`: use the lower 20-hour ceiling until this discrepancy is resolved. Deduct final measured usage from all prior jobs and commitments from any running allocations before submission; a historical ledger snapshot is insufficient. The four-hour Slurm limit is a maximum within that remaining budget, not an additional allowance or expected runtime. Shorten it to the actual remaining allowance; use one GPU. L4 is the default; another available single-GPU partition can be selected with `sbatch --partition=...` when permitted by the current resource policy.

Run CPU stages on an appropriate CPU allocation. Replace the paths below with existing persistent HPC directories; no downloads are performed.

```bash
export SOURCE=/persistent/project/pilot/results/formal-l4-JOB/tminus
export OUT=/persistent/project/pilot/results/next-round-JOB
export PROJECT_DATA_DIR=/persistent/project/pilot/data
export PYTHON=/persistent/project/.venv/bin/python
"$PYTHON" pilot/next_round.py preflight --source "$SOURCE" \
  --data-dir "$PROJECT_DATA_DIR" --output "$OUT"
```

Preflight checks source split separation, model ID, checkpoint dimensions against the pinned encoder's configured projection dimension, staged images, and the cached encoder. It hashes the inputs and current model snapshot, and exports `semantic_audit.csv`: a reproducible 100-pair sample stratified by intervention and candidate count. The original checkpoint metadata lacks an exact encoder revision, so original training-time revision identity cannot be proven; the follow-up pins and hashes the currently staged snapshot.

Before reserving a GPU, also inspect historical center-crop geometry on CPU:

```bash
"$PYTHON" pilot/visibility_audit.py --manifest "$SOURCE/manifest_snapshot.json" \
  --data-dir "$PROJECT_DATA_DIR" --output "$OUT/visibility_audit.json"
```

This records target/region visible fractions and cases in which the historical spatial baseline used a nearest-patch fallback. It audits that implementation's continuous 224-pixel / 14-patch geometry; it does not claim pixel-exact preprocessing or rerun spatial predictions. Use the output to identify invalid crop-versus-patch comparisons before further attribution experiments.

## Semantic audit

Inspect each image, target box, other candidate regions, and both captions. Fill `target_change_valid`, `controls_preserved`, and `grammar_acceptable` with `yes`, `no`, or `uncertain`; fill `audit_decision` with `valid`, `invalid`, or `uncertain` and supply `reviewer`. For deletion/natural captions, check that the target is actually omitted, including synonyms and implicit mentions. For matched generalization, check that intended target detail is removed while the generalized entity remains meaningful. Check every other candidate's coverage stays unchanged. Record multiple omissions, changed relations, invisible targets, or ambiguous coverage in `notes`.

No row is automatically approved. Existing audit edits are preserved on repeated preflight. Summary eligibility requires all three checks to be `yes`, `audit_decision=valid`, and a nonempty reviewer. The audit is a diagnostic sample, not a population prevalence estimate.

## GPU extraction and frozen inference

```bash
# Use a smaller time limit if less budget remains. This example caps this job at one hour.
export LIMIT_SECONDS=3300
sbatch --partition=g2-standard-12 --time=01:00:00 hpc/next_round.sbatch
```

Monitor `gpu-JOB.log` and `utilization-JOB.log` in `$OUT`. GPU, CPU, job state, elapsed time, and recent progress are recorded every five seconds. Diagnose persistent low utilization or stalled progress promptly; cancel if no productive in-scope work is ready. This script performs extraction and inference only; CPU bootstrapping occurs after GPU release. It combines copied intervention rows for one `run.features` call, deduplicating shared crops, texts, and phrase embeddings with one encoder load. Each frozen scorer is loaded once for all interventions, and predictions are sliced back into per-intervention files. The original source rows remain unchanged. Serial image preparation remains an optimization opportunity. Do not claim utilization is fixed solely because monitoring is present. Completed feature caches survive a retry; interrupted in-progress extraction may need repeating.

The script fails if source artifacts or images change after preflight, the current encoder differs from source model metadata, or feature/checkpoint dimensions do not match. Use a new output directory for changed inputs. It never trains or selects another checkpoint.

## CPU summary

```bash
"$PYTHON" pilot/next_round.py summarize --output "$OUT"
sacct -j JOB_ID -X --format=JobID,Partition,Elapsed,State,AllocTRES
```

`summary.json` reports each intervention on all available pairs, the common pair intersection, confirmed-valid audit rows, and separate candidate-count strata. It includes tie-aware delta and absolute Acc@1/MRR, target delta, off-target drift, normalized target delta and local contrast, image bootstrap intervals, and paired differences against inverse cosine. The constant-score baseline exposes the chance level within each stratum. Normalized deltas divide by each pair's mean absolute delta, with values at or below `1e-8` mapped to zero. Interpret normalized scores together with absolute signal strength; they do not independently establish locality.

Natural-caption delta/locality metrics remain exploratory until other-region coverage is confirmed. Absolute ranking still assumes that the designated target is the intended omitted target; multiple-omission cases need multi-label evaluation later. The bootstrap conditions on three existing seeds. The summary distinguishes `seed_metric_mean` from `prediction_ensemble` and does not imply a new independent test set.

Do not commit images, feature caches, model weights, or predictions. Copy only reviewed compact summaries and provenance into a tracked research report after checking results.
