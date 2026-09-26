# Project memory

## HPC utilization policy

- Follow the cumulative budget and handoff reconciliation in `GOAL.md`; never treat the 300-hour course quota as authorization.
- Use any currently available suitable GPU, including L4 or A100. Start with one device, record its type, and adapt model size, resolution, batching and staged loading to measured capacity and throughput. Stage datasets, model weights, environments, and CPU preprocessing before requesting compute.
- Do not leave an allocated GPU or CPU node idle or at persistently low utilization. Keep the accelerator fed with suitable batching, workers, caching, and overlap between input preparation and inference.
- Maintain a queue of productive fallback GPU tasks (feature caching, smoke tests, baselines, ablations, or evaluation) and switch to the next valid task if the primary run stalls. Every fallback must produce an artifact needed by the stated experiment.
- Never run synthetic burn loops or unrelated workloads merely to inflate utilization. Keep a finite queue of productive tasks, but **do not manually release, cancel, or `scancel` any CPU/GPU allocation**. The user's latest instruction is to let the server reclaim allocations itself because manual release interrupts the working session.
- Emit frequent progress and utilization records for long jobs. Monitor `nvidia-smi`, CPU utilization, job state, elapsed time, and the experiment log after submission.
- If utilization stays unexpectedly low, diagnose promptly and move to another useful in-scope task when one is ready. Do not cancel or restart the allocation; preserve checkpoints and let the scheduler handle reclamation.
- Checkpoint and cache reusable work so a preempted or cancelled job can resume without repeating expensive computation.
- Record measured Slurm usage and utilization issues in the compute ledger and `research/AGENT_LOG.md`.
- Sample utilization and progress every 5 seconds. During steady GPU-ready work, target 70–90% where feasible; diagnose <30% for two minutes or stalled progress. These are internal targets, not verified cluster eviction rules.
- Subagents may prepare independent code, data or analysis when useful; one main owner controls job submissions, output paths and cumulative budget.
- Keep the complementary-visual-information research objective; individual method failures require diagnosis and redesign, not automatic abandonment of the question. Follow the autonomous failure-to-redesign loop and independent-confirmation gates in `GOAL.md` toward positive results and novelty. Preserve negative evidence and distinguish plans from completed results; never manufacture a positive claim or repeatedly tune against a confirmation set.
