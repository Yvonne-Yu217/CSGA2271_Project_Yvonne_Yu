# Project memory

## HPC utilization policy

- Treat the 24 GPU-hour phase budget in `GOAL.md` as a hard cap.
- Use exactly one L4 (`g2-standard-12`) for formal runs. Stage datasets, model weights, environments, and CPU preprocessing before requesting it whenever possible.
- Do not leave an allocated GPU or CPU node idle or at persistently low utilization. Keep the accelerator fed with suitable batching, workers, caching, and overlap between input preparation and inference.
- Maintain a queue of productive fallback GPU tasks (feature caching, smoke tests, baselines, ablations, or evaluation) and switch to the next valid task if the primary run stalls. Every fallback must produce an artifact needed by the stated experiment.
- Never run synthetic burn loops or unrelated workloads merely to inflate utilization. If no productive in-scope task is ready, release the allocation immediately and resume after staging is complete.
- Emit frequent progress and utilization records for long jobs. Monitor `nvidia-smi`, CPU utilization, job state, elapsed time, and the experiment log after submission.
- If utilization stays unexpectedly low, diagnose promptly. Cancel or restart a stalled/idle job rather than waiting for the scheduler to reclaim it.
- Checkpoint and cache reusable work so a preempted or cancelled job can resume without repeating expensive computation.
- Record measured Slurm usage and utilization issues in the compute ledger and `research/AGENT_LOG.md`.
