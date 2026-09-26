# Goal: Validate the proposal globally, then decide whether to continue

Updated: 2026-09-26. The latest user instructions supersede earlier device restrictions.

## First objective

Evaluate the original proposal's full set of claims against valid evidence, within the remaining authorized compute budget. Establish what works, what fails, and what remains untested. A credible negative result is a valid outcome. Do not require an MLP improvement, force a CVPR narrative, or silently count unrun experiments as complete.

The learned scorer currently trails inverse cosine on deterministic deletion: 77.22% versus 84.50% Acc@1; paired difference −7.28 percentage points, 95% image-bootstrap CI [−11.70, −3.33]. This is a mean of seed metrics, not a prediction ensemble. Natural-caption and SigLIP runs used separate training and do not establish frozen-model transfer.

## Evidence and completion gates

1. **Recover and correct evidence (CPU).** Recover HPC manifests, predictions, checkpoints, model metadata and final Slurm accounting. Preserve historical metrics; distinguish seed averages from ensembles, independent retraining from transfer, and similarity proxies from recoverability. Fix cache provenance and tie handling before new conclusions.
2. **Validate the task.** Export and manually review 100 stratified intervention pairs. Record target fact granularity, whether it is visible and genuinely omitted, and which other regions change. Natural captions cannot assume every non-target region remains covered. Mark invalid/ambiguous cases; do not invent human annotations. Freeze a reviewed diagnostic subset before comparing methods.
3. **Run the bounded follow-up.** Reuse deletion-trained checkpoints without optimization on matched/natural captions. Report same-caption absolute ranking, intervention-difference ranking, tie-aware metrics, three seeds, a separately identified prediction ensemble, paired image-bootstrap intervals and scale-normalized changes. The existing test set is exploratory after repeated inspection.
4. **Decision gate.** If labels cannot support the operational definition, repair or replace the task. If strong similarity/phrase baselines remain best, finish a rigorous course comparison and failure analysis, or recommend changing direction. At most two motivated learned variants may be proposed after the validity gate; no unbounded tuning.
5. **Conditional confirmation.** Only with a meaningful validation signal, preregister a new untouched image subset, strong direct semantic-coverage baselines, automatic-region visibility/recall controls, and an independent reader evaluation. Visual Genome transfer, independent semantic coverage, and a complete official attribution reproduction remain unverified. They cannot be marked complete from the current pilot.
6. **Deliver.** Maintain corrected proposal, review, runnable staged experiment scripts, report, compute ledger and Agent Log; push code and compact evidence. Keep data, caches and checkpoints outside Git. Publication expansion has its own gated proposal.

See `research/RESULTS_REVIEW_AND_NEXT_STEPS.md`, `research/CVPR_EXTENSION_PROPOSAL.md` and `hpc/NEXT_ROUND.md` for the decision tree and execution details.

## Compute budget and hardware

- Course quota is 300 GPU hours; it is **not** the phase authorization. The imported HPC record states a revised 24 GPU-hour cap; the earlier local conversation authorized 20. Reconcile the handoff and final `sacct` ledger before submission; if unresolved, use the lower 20-hour total. Count previously consumed allocations, concurrent jobs and GPU count. No automatic reset or quota extension.
- The recorded 1.1267 GPU hours is a snapshot with a running job, not final usage. Next diagnostic work has an internal ceiling of **4 additional allocated GPU hours**, bounded further by the actual remaining authorized balance. This is a maximum, not a request to consume it.
- Use available compatible hardware: L4, A100, or another suitable GPU. Do not wait for A100 as a requirement. Default to one GPU; batch size follows memory. Multi-GPU only if measured throughput and total allocation cost justify it within the same budget.
- Prepare downloads, environment, semantic audit, CPU reports and commands before allocating a GPU. Use batch jobs to survive terminal disconnection. Cluster idle-reclamation behavior must be checked locally; utilization alone does not guarantee a session survives.

## Productive utilization is required

Persistently low GPU utilization delays the project and must be investigated promptly. During steady GPU-ready work, target 70–90% utilization where the workload permits. These are internal diagnostic targets, not claimed cluster rules. Record device utilization, memory, timestamps and progress every 5 seconds. If utilization remains below 30% for two minutes, or progress stalls, check input loading, batching, CPU synchronization, I/O and memory pressure. Separate warmup, downloads and CPU-only intervals from compute-stage statistics.

Maintain a finite queue of needed extraction/inference tasks. Cache reusable features, batch text/crops, prepare inputs on CPU, and checkpoint completed stages. A lightweight cached-feature scorer or bootstrap belongs on CPU if GPU work is too small. Necessary subagents may prepare code, audit outputs or stage independent work; the main agent controls submissions, shared outputs and cumulative budget. Never duplicate experiments just to raise utilization. Release the node when no useful in-scope GPU task is ready.

## Proposed conversation goal text

`/goal 第一目标：在剩余已授权预算内验证 proposal 全局内容，先审计现有结果与标签，再运行已准备好的冻结模型迁移和有效性实验，根据证据决定保留、收缩或更换方向，不强行证明原假设。使用任何可用且合适的 GPU，不限定 A100/L4；提前准备代码与数据，持续监测并优化有效 GPU 占用，必要时用 subagent 并行准备工作。保持预算账本，更新 proposal、实验计划、报告和 repo；CVPR 扩展必须通过独立决策门槛。`

This file updates the repository goal. It does not assert that the conversation-level goal was changed; the available goal API cannot rewrite an existing goal's objective.
