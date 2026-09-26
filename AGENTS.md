# Project memory

## HPC utilization policy

- Follow the cumulative budget and handoff reconciliation in `GOAL.md`; never treat the 300-hour course quota as authorization.
- Use any available compatible GPU, including L4 or A100; neither is mandatory. Prefer one GPU unless measured throughput justifies more within the same budget. Stage datasets, model weights, environments, and CPU preprocessing before requesting it.
- Do not leave an allocated GPU or CPU node idle or at persistently low utilization. Keep the accelerator fed with suitable batching, workers, caching, and overlap between input preparation and inference.
- Maintain a queue of productive fallback GPU tasks (feature caching, smoke tests, baselines, ablations, or evaluation) and switch to the next valid task if the primary run stalls. Every fallback must produce an artifact needed by the stated experiment.
- Never run synthetic burn loops or unrelated workloads merely to inflate utilization. If no productive in-scope task is ready, release the allocation immediately and resume after staging is complete.
- Emit frequent progress and utilization records for long jobs. Monitor `nvidia-smi`, CPU utilization, job state, elapsed time, and the experiment log after submission.
- If utilization stays unexpectedly low, diagnose promptly. Cancel or restart a stalled/idle job rather than waiting for the scheduler to reclaim it.
- Checkpoint and cache reusable work so a preempted or cancelled job can resume without repeating expensive computation.
- Record measured Slurm usage and utilization issues in the compute ledger and `research/AGENT_LOG.md`.
- Sample utilization and progress every 5 seconds. During steady GPU-ready work, target 70–90% where feasible; diagnose <30% for two minutes or stalled progress. These are internal targets, not verified cluster eviction rules.
- Subagents may prepare independent code, data or analysis when useful; one main owner controls job submissions, output paths and cumulative budget.
- Do not force the original proposal to succeed. Follow the validity and direction-change gates in `GOAL.md`; preserve negative evidence and distinguish unrun work from completed results.
