# L4 resource and productive-execution plan

Updated 2026-09-26 after HPC commit `2f96409` and the user's fixed-L4 instruction. Authority: [GOAL.md](../GOAL.md). The old pilot's resource envelopes and device alternatives do not govern this plan.

## Hardware decision: fixed L4

**All subsequent project GPU experiments use NVIDIA L4. Start with one L4 on `g2-standard-12`.** Adapt model size, precision, resolution, batch size and training strategy to this choice. There is no automatic device-type upgrade or availability-based substitution.

The [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/) list 24GB memory and 300GB/s memory bandwidth. Node-visible usable memory and other processes must be measured on the actual allocation.

### What the repository already demonstrates

| Recorded workload | L4 measurement | Interpretation |
|---|---|---|
| 1,400 Qwen2.5-VL-3B crop observations | 455.72 seconds; 7,721.54 MiB peak torch allocation | Existing frozen observation workload fits on one L4 |
| 500 full-image completions, default image resolution | 180.09 seconds; about 7.55 GiB peak torch allocation | Strong baseline fits; poor crop results are not evidence of insufficient GPU memory |
| 500 full-image completions, 50,176-pixel cap | 140.64 seconds; about 7.39 GiB peak torch allocation | Useful reference for bounded-resolution inference |
| Separate Qwen focus-ambiguity prompt workloads | About 96.6 seconds; 10.79 GiB peak torch allocation | Memory varies materially with input/configuration |

Sources: [E0–E2 progress](../research/E0_E2_PROGRESS.md), [agent log](../research/AGENT_LOG.md), [focus preflight](../research/FOCUS_AMBIGUITY_PREFLIGHT.md). These are historical measurements, not profiling of the redesigned global-plus-region method. Torch allocated memory excludes some device/runtime use and reserved buffers; do not equate it with total VRAM usage.

**Conclusion:** L4 is sufficient for the already-run 3B inference and is a justified platform for the next controlled diagnostics. The new workloads and later training must be made to fit and profiled; their exact speed and memory are not yet known.

## Configuration for the next round

- Retain the pinned Qwen2.5-VL-3B observer for controlled attribution. Use its [version-specific official model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for multi-image, batching and pixel controls; pin dependencies rather than blindly install the latest version.
- Cache grounding separately, then unload that experiment's model and load the observer **within the same retained allocation**. Optional segmentation is a later diagnostic stage, not a mandatory extra resident model.
- Bound total global-plus-ROI visual tokens and output length. Record processed dimensions and visual-token counts, not just the original crop area.
- Profile a finite batch sweep such as 1/2/4/8 on real representative development inputs; stop increasing when memory/latency/throughput worsens. Choose the highest useful throughput with safe headroom, initially leaving approximately 10–15% of device memory available and verifying no upward drift. This is a tuning target, not a guarantee.
- Bucket similar visual-token and text lengths to limit padding. Predecode/prefetch inputs on CPU, cache crops/features, and overlap preparation with inference where supported.
- Select efficient attention only if the pinned environment supports it and output behavior is checked in the authorized implementation-validation stage. Do not break the working environment during an allocation to chase a theoretical speedup.
- For later learned policies, start with cached-feature cross-attention or small adapters; profile activations, optimizer state and sequence lengths before training. If a design is too large, revise model/configuration on L4 and document its limitations.
- Begin with one GPU-owning process. Do not launch multiple heavy models merely because several subagents are available. Additional concurrent processes require measured device headroom and a net gain in completed useful work.

## Continuous utilization and progress monitoring

Keeping allocated GPUs productive is a project priority and must appear in every HPC handoff. A high `nvidia-smi` peak or high memory occupancy alone is insufficient.

During active execution, a separate monitor should sample every **5 seconds**:

- UTC timestamp, job ID, run ID and device ID;
- GPU utilization, device memory used/total, and available power/clock/throttling indicators;
- CPU load, preprocessing queue depth or I/O wait when available;
- completed/remaining examples, most recent output/checkpoint time and valid outputs/second;
- rolling utilization, idle time, model-loading time and allocation elapsed time.

During steady GPU-ready work, target **70–90% or higher when it produces useful throughput**. Investigate <30% for two minutes, a stale heartbeat or no new artifacts. These are internal operating thresholds, not verified cluster eviction rules. Short loading/checkpoint/CPU transitions can dip; do not claim physically constant utilization.

### Response to low utilization

| Symptom | Productive action inside the existing allocation |
|---|---|
| CPU decode/I/O queue empty | Increase appropriate workers, prefetch staged files, cache/precompute views, inspect storage contention |
| Small batches with memory headroom | Increase token-bucketed batch size after bounded profiling |
| Memory-bound configuration/OOM | Reduce batch/total visual tokens, stage models, preserve outputs and resume smaller configuration |
| GPU busy but artifacts stalled | Inspect process/error logs and heartbeat; identify compute versus deadlock; preserve checkpoints |
| Current finite job completes | Start the next dependency-satisfied, manifest-approved task |
| Waiting on human labels | Run only already-approved independent inference/preparation; do not claim the label gate passed |
| No valid GPU-ready task remains | Log the blocker, advance CPU/code/data preparation, preserve allocation for server-managed reclamation |

Never use synthetic burn loops, unrelated workloads, duplicate finished runs or held-out inference to hide idle time.

## Task preparation and subagents

Follow [METHOD_REDESIGN_PLAN.md](../research/METHOD_REDESIGN_PLAN.md) for the finite Q0–Q5 queue. Prepare the code/configuration, inputs, model revisions, output location and resume command for the next valid task while the current one executes.

Subagents may independently implement scoped modules, inspect literature, prepare data or review results. One owner controls all GPU submissions, device memory, shared outputs and the cumulative compute ledger. Subagents do not independently claim an allocation or load competing models onto the same L4.

No new queue runner or monitor implementation is claimed by this document. The proposed modules must be implemented and prepared before the next execution stage; existing scripts only reproduce their recorded workloads.

## Preserve allocations and session state

**Do not manually release, cancel, `scancel`, or restart CPU/GPU allocations.** The server handles reclamation. Do not destroy a working session, environment, cache or checkpoint to address poor utilization.

It is normal to end a completed experiment process and unload its model before the next stage while keeping the allocation/session intact. A stalled application can be diagnosed and recovered within the retained allocation with saved progress; allocation cancellation is not the recovery mechanism.

## Budget and measured accounting

The user removed the old pilot-hour ceiling as a research-design constraint, but did not authorize consuming all 300 course GPU-hours. Before any submission, reconcile the current authorized balance with actual Slurm usage and running allocations. Historical 20/24-hour envelopes and the 2.0333-hour snapshot are not a verified current balance.

Record one row per unique allocation/job and avoid double-counting parent/step or typed/generic GPU resource fields. Track GPU count × allocated elapsed hours, work completed, monitoring gaps and idle overhead. Apply the cluster's actual charging rules if they differ; do not infer them from utilization.

Historical jobs 1979/1998 appearing in records are not proof of current running state. This local revision did not connect to HPC, query live `nvidia-smi`/`sacct`, launch work, or release any allocation.
