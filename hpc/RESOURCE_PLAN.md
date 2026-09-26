# HPC resource plan after the initial results

The current authority is `GOAL.md`; historical L4 launchers reproduce the earlier experiment. Their defaults do not impose a device restriction on future work.

| Partition supplied by user | Hardware | Use |
|---|---|---|
| `n2c48m24` | CPU | Staging, audit, cached-feature analysis and report |
| `g2-standard-12` | 1 L4 | Suitable default for frozen encoder extraction |
| `g2-standard-24` | 2 L4 | Only if measured benefit justifies allocated GPU hours |
| `c12m85-a100-1` | 1 A100 40GB | Equally acceptable if available |
| `c24m170-a100-2` | 2 A100 40GB | Not required by the next round |
| `interactive` | Site-defined interactive service | Inspect site configuration before use |

Account: `ds_ga_1006_001-2026fa`. Confirm GRES spelling, memory, wall-time limits and billing locally. Use the new staged workflow in `hpc/NEXT_ROUND.md`; prepare all inputs before submitting a GPU job. Batch execution reduces dependence on an interactive terminal connection.

## Budget

The course quota of 300 GPU hours must not be used as the project budget. The imported HPC log records a 24-hour revised cap, while the earlier local authorization was 20. Reconcile before submission, using 20 total if unresolved. Deduct final `sacct` usage, including running and concurrent jobs. The historical 1.1267-hour ledger is incomplete. Next diagnostics have an internal upper bound of 4 additional allocated GPU hours, or remaining authorized balance if smaller. Device-specific charging, if any, also applies.

## Stages and stop conditions

1. CPU: recover artifacts, preflight local model/data/checkpoints, export semantic review, resolve reporting caveats and finalize commands.
2. GPU only as needed: cache missing embeddings and infer from frozen deletion-trained checkpoints. Use existing compatible caches when available. One GPU is sufficient; adapt batch size to its memory.
3. CPU: bootstrap, normalization, tables and decision report. Do not retain GPU allocation for manual review or report writing.
4. Stop if evidence is invalid or no meaningful progress is possible. Do not launch model variants, Visual Genome or new direction experiments before their decision gate.

P0–P2 are complete. Job 1972 performed the frozen inference in 683 allocated seconds and was released after output generation; the five-second trace captured a 91% utilization peak. The semantic gate failed (44 valid / 53 invalid / 3 uncertain in a Codex visual diagnostic, with zero human-confirmed rows), so stage 4 applies: no additional GPU experiment is currently authorized. Notebook allocations created for CPU reporting are released rather than filled with duplicate work.

For GPU-ready stages, seek 70–90% utilization where feasible, monitor every 5 seconds, and diagnose sustained <30% for two minutes or missing progress. These thresholds are internal targets, not scheduler rules. Improve batching/workers/I/O, move small CPU work off GPU allocations, or release the node. Keep useful fallback tasks ready, never synthetic load. Subagents may prepare independent work; one owner controls submissions and budget.

## Original proposal coverage

| Component | Current evidence | Remaining limitation |
|---|---|---|
| Learned score / RQ1 | Three seeds and strong baselines | Superiority unsupported |
| Intervention response / RQ2 | Three intervention families plus frozen deletion-checkpoint transfer | Semantic diagnostic validity is weak; no human confirmation |
| Generalization / RQ3 | Separately trained SigLIP sensitivity | Not weight transfer; Visual Genome unrun |
| Data | 350 images / 1,165 pairs, split/hash audit, 100-row Codex visual review | Only 44/100 diagnostic rows valid; no human labels |
| Spatial attribution | Task-adapted CCI, Grad-ECLIP, patch methods; crop visibility audited | 11/342 regions invisible and 141 partial; not full official reproduction |
| Single-caption localization | Tie-aware frozen summary completed | Existing set is exploratory; audited subset is small |
| Recoverability | Same-CLIP masked similarity | No independent reader or new-fact measurement |
| Automatic regions | Patch cluster sensitivity mapped to boxes | Independent region recall/quality unverified |

Preserve historical results and report these limitations explicitly.
