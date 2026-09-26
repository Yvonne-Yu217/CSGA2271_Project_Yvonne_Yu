# Method redesign and HPC handoff after 2f96409

Date: 2026-09-26. Status: **planned, not implemented or executed**. Authority: [GOAL.md](../GOAL.md) and [current proposal](FOLLOWUP_PROPOSAL.md).

## Research decision

Retain the complementary-visual-information objective. Replace the failed observation design, isolate its defects, and continue method development until independent positive evidence and a defensible contribution meet the goal. The focus-ambiguity branch is archived; its latest paired proposal is not a completed positive experiment.

The immediate task is R0–R2: valid evidence, controlled geometry/context comparisons, and residual headroom over a strong full-image baseline. Do not launch selector training merely to keep a GPU busy.

## Evidence audit

| Evidence | Interpretation for implementation |
|---|---|
| `acquisition/prepare_e0.py::grid_candidates` produces 9 tiles, 4 halves and a full image | There was no segmentation model in this acquisition experiment. Retain grids as a baseline; add actual automatic grounding separately. |
| `acquisition/observe.py::DEFAULT_PROMPT` is caption-independent | A new context-conditioned observer needs new inputs, caches and labels. |
| `acquisition/full_image_completion.py::PROMPT` sees the caption and requests mentioned-entity details | Match task instructions before attributing differences to crop geometry. |
| Full-image strict proxy 48.6% versus old oracle 30.8%, paired difference 17.8pp [9.2, 26.2] | Do not train on the unchanged pipeline. This is automated evidence against the package, not a proof that the research question fails. |
| Qwen supports all 1,281 non-empty claims; SmolVLM supports 393 | Complete independent human review; neither judge is ground truth. |
| Crop-verification screen uses full-image model judgments as reference | Keep the historical rejection, but call this cross-view disagreement rather than human-confirmed falsehood. |
| Stored `pixel_cost` is source-box area | Add processed visual tokens and full-path runtime. Count montage/planner and proposer inputs. |

Source results: [provisional screen](PROVISIONAL_DIRECTION_SCREEN.md), [metrics](provisional_direction_metrics.json), [E0–E2 progress](E0_E2_PROGRESS.md). No new quantitative outcome is asserted here.

## P0: CPU preparation and review

1. Freeze the image-disjoint development manifest, old-data exclusion list, duplicate groups and model revisions. All previously inspected samples are development data.
2. Reuse the method-hidden stopping audit to diagnose the previous result, plus a random sample to estimate general label error. Keep two independent human reviews and adjudication distinct from automated checks.
3. Record separate labels for visible evidence, instance identity, text coverage, crop completeness and generated-claim validity. Unknown is a valid state; do not invent human labels to pass E0.
4. Stage images and model weights before inference. Generate source-image dimensions, neutral candidate coordinates, padded boxes and relation unions using model-visible inputs only.
5. Predeclare the R1 matrix, budgets, scoring and finite queue. Freeze the new R2 thresholds and sample plan before observing its results.

## P1: modules to implement before HPC execution

The following names are **proposed deliverables**, not existing runnable commands.

| Proposed component | Required behavior and acceptance evidence |
|---|---|
| `acquisition/propose_grounded_regions.py` | Pin a local Grounding DINO model; derive queries from available text/visible scene information; save boxes, confidence, source hash and query provenance; handle no detection without treating it as factual absence. |
| `acquisition/build_evidence_views.py` | Create unmasked entity boxes, fixed padding variants, relation-union views and optional mask views; include full image and STOP; record original-to-processed coordinates; cap/merge candidates with a preregistered label-free rule. |
| `acquisition/observe_conditioned.py` | Provide the same T, task instruction, frozen backbone and output format to all relevant arms; support crop-only, global-only and global-plus-ROI inputs; explicit same-image relationship; resumable per-context cache. |
| `acquisition/run_redesign_matrix.py` | Execute the frozen R1 manifest; include matched full-image and repeated-call controls; reuse only truly identical inputs; expose run fingerprints and progress. |
| `acquisition/evaluate_residual.py` | Separate raw evidence coverage from generated facts; compute residual gain over a shared first full-image completion; preserve instance IDs, errors, duplicates, unknowns and image-cluster uncertainty. |
| `hpc/run_manifest_queue.py` | Run only finite approved ready tasks inside the retained allocation; dependency checks, unique output paths, cache integrity and checkpoint/resume; never submit surprise jobs or cancel the allocation. |
| `hpc/monitor_progress.py` | Independent five-second GPU/memory/progress sampling, rolling utilization and stale-heartbeat alerts; report low utilization without generating artificial load. |

Extend the existing schema instead of bypassing its gold/public separation. The current `acquisition/schema.py` does not automatically support every new input mode. Audit changes to candidate visibility, context-dependent observations and derived action utilities before claiming compatibility.

New cache keys must include image SHA, context hash, candidate geometry/instance IDs, global-view configuration, model and processor revision, prompt hash, decoding settings and output schema revision. In R2, include the initial completion B's content/hash, generator revision and prompt, and pass the identical T+B state to every second-step observer, selector and fact filter. B is a prediction, not gold; its errors remain scored. A context-independent caption cache cannot be silently relabeled as context-conditioned output.

## P2: single-GPU productive execution queue

Use any currently available suitable GPU, starting with one device and recording its exact type. Adapt each workload to measured memory and throughput. Do not manually cancel or release any existing allocation.

| Queue stage | Real artifact produced | Concurrent preparation |
|---|---|---|
| Q0: representative profiling | Bounded batch/resolution measurements using genuine development inputs; retain valid outputs | CPU workers prepare length/vision-token buckets and the remaining input shards |
| Q1: automatic grounding | Cached boxes, query provenance and failure cases | CPU builds padding/union views and review materials from completed shards |
| Q2: full-image controls | Direct completion, structured facts, repeated/full-image controls | A subagent reviews the next code/configuration change in a separate file scope |
| Q3: R1 observation matrix | Caption/context/geometry ablation outputs | CPU assembles blinded fact-review packets and validates cache completeness |
| Q4: independent automated sensitivity | Cross-model entailment/support predictions with unknowns and judge provenance | CPU computes descriptive diagnostics; humans complete fact review |
| Q5: approved R2 actions | Residual-value outcomes and natural caption-state comparisons | Prepare the next evidence-based design, conditional on the current gate |

Q4 is a sensitivity analysis, not a substitute for human truth. R2 development screening can be prepared while review is pending, but formal gate passage and selector training require valid labels. Optional masks are queued only when R1 supports a need to test them; they are not a default way to fill time.

The owner records each task's `run_id`, dependency status, input/output hashes, planned useful artifact, estimated/actual duration, device memory profile and authorization balance. Keep at least the next valid GPU task staged while the current one runs. Do not inflate the queue with reruns of completed caches.

## R1 design details

- Geometry: fixed grids; automatic entity boxes; padded entity boxes; relation-union regions.
- Text: caption-independent versus the same caption-conditioned completion instruction.
- Context: region-only versus global thumbnail plus the same region.
- Optional diagnostic: masks and manually annotated evidence regions, reported separately from deployable automatic proposals.
- Mandatory controls: one full-image completion; higher-resolution full image; equal-call full-image refinement; duplicate full-image view; full-image structured extraction plus NLI filtering; strong prompted crop planner.

Use a shared instruction template for the factorial arms, changing only the declared factors. Gold regions are upper bounds only. Use deterministic decoding initially; if stochastic outputs are later studied, fix and report all seeds. Match final output length, report complete costs, and include an equal-total-vision-budget comparison where feasible.

The no-caption factorial arm removes T from the observer only, while holding its candidate pool fixed. Caption-derived grounding can still convey textual information through the candidates. Do not call this a fully text-independent system; add a separate image-only proposer/planner and selection control for end-to-end caption necessity.

Suggested initial development scope: 200–300 new images with a detailed 100-image annotation subset, starting with a frozen smaller profiling block. All states and candidates from one image stay together. Limit the initial candidate pool to a fixed cap, for example 16 actions including full image and STOP, and fix how candidates are deduplicated and selected before labels are read. Vary the cap only in a declared coverage–cost study.

## R2 decision rules

1. If the matched global-plus-region setup improves evidence access, measure remaining coverage gain over strong full-image controls, including an equal-call second full-image pass.
2. If manual-region upper bounds succeed but automatic candidates fail, fix grounding and binding rather than train an action policy on poor candidates.
3. If no region family offers residual headroom, evaluate the full-image fact-selection family with strong extraction/NLI and prompting baselines. Keep the same complementary-facts objective.
4. If perception limits every family, compare resolution or observer strength on L4 with the selection rule held fixed, then redesign the method to fit the observed capability.
5. If judge gains vanish under independent review, repair labels and the evaluator. No training gate passes on the disputed proxy.

Before R2 results, freeze the practical effect threshold, error noninferiority margin, sample size and strongest comparator. The proposal suggests about 5pp oracle headroom and later about 3pp learned gain, but these are planning thresholds, not forecasts or permission to reclassify old failures.

## Mandatory per-round record

Create a separate record for each new design before inference, retaining every subsequent result:

```yaml
round_id: R1-v1
status: planned
parent_failure_record: research/PROVISIONAL_DIRECTION_SCREEN.md
objective: additional correct visual facts given existing text
bottleneck_hypothesis: to_be_frozen_before_run
material_method_change: to_be_frozen_before_run
primary_sources_and_code_revisions: []
development_manifest_hash: required
confirmation_manifest_hash: locked_separately
baseline_and_cost_controls: required
primary_metric_and_effect_threshold: required
error_margin_and_uncertainty_unit: required
gpu_type: L4
ready_queue_manifest: required
authorization_and_usage_ledger: required
results_and_failure_examples: pending
decision_and_next_design: pending
```

After a failure, identify the bottleneck, review relevant papers/code, and write the next materially different design. Two failed variants aimed at the same bottleneck trigger a method-family review. Continue in authorized resources; unresolved access/annotation/budget constraints must be reported, not bypassed.

Fresh confirmation data does not by itself control repeated testing. Before the first formal attempt, register a project-wide confirmation limit and alpha budget spanning method revisions. Initial plan: no more than two attempts at two-sided alpha 0.025 each, with 97.5% intervals for the primary gain; account for any additional primary comparisons. Log every attempt and do not reset the budget by renaming a method. Further rounds remain development when the confirmation budget is exhausted.

## Handoff invariants

- Current stage is CV validation, not finance and not a successful E3 method.
- Any suitable available GPU may be used; record the exact device and configuration for every run.
- Observe utilization/progress every five seconds during HPC execution; investigate <30% for two minutes or stalled progress. Aim for high sustained useful throughput, not artificial utilization.
- Subagents prepare independent tasks concurrently; a single owner controls GPU submissions and total spend.
- Preserve allocations, sessions, caches and checkpoints. No manual `scancel`, release or allocation restart.
- No live HPC state was checked in this local documentation update. Do not infer that historical job IDs are still running.
