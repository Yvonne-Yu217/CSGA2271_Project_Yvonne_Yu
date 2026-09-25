# NYU HPC resource and execution plan

## Request now

Based on the partitions and quota provided by the user:

| Resource | Initial request |
|---|---|
| Account | `ds_ga_1006_001-2026fa` |
| Partition | `c12m85-a100-1` |
| GPU | 1 × NVIDIA A100 40GB |
| CPU | 8 cores initially; up to 12 if useful |
| Host memory | 64 GB |
| Initial wall time | 8 hours, after data/model staging |
| Persistent/scratch capacity | 200 GB preferred; increase if retaining full raw datasets and many caches |
| Software | Python 3.11/3.12, CUDA-compatible PyTorch, Transformers; Git and outbound data access or a staged transfer |

A100 40GB is sufficient for the planned frozen CLIP/SigLIP workflow and sequential attribution/region-model inference. Final peak memory must be measured. A single `g2-standard-12` L4 is a viable fallback with smaller batches; memory and time settings must match that partition's actual limits. Two GPUs are useful later for independent runs, but no distributed-training dependency is needed.

Use CPU partition `n2c48m24` for downloads, extraction, annotation validation, and cached-feature small-scorer training. Do not hold a GPU while waiting for data. Exact account permissions, allowed wall times, GPU GRES spelling, and charge multipliers are not yet verified on the cluster.

Example allocation, using standard Slurm syntax; confirm site settings first:

```sh
sinfo -p c12m85-a100-1 -o '%P %l %G %m %c'
salloc --account=ds_ga_1006_001-2026fa \
  --partition=c12m85-a100-1 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00
srun --pty bash -l
```

Sources for generic flags: [Slurm salloc](https://slurm.schedmd.com/salloc.html), [Slurm sbatch](https://slurm.schedmd.com/sbatch.html). Partition/account values come from the user, not these generic documents.

## Planning envelope, not measured runtime

| Work package | GPU-hour ceiling to plan against |
|---|---:|
| Environment, throughput and end-to-end smoke experiments | 8 |
| CLIP/SigLIP feature extraction on planned datasets | 40 |
| Scorer runs and loss/feature/intervention ablations | 24 |
| Decomposition, Grad-ECLIP and CCI comparisons | 64 |
| Automatic regions and Visual Genome transfer | 48 |
| Recoverability/faithfulness experiments | 32 |
| Essential reruns and final figures | 24 |
| **Initial working envelope** | **240** |
| **Unallocated reserve** | **60** |

These are scheduling caps, not evidence that every run will finish inside them. After the first 100–500 images, estimate `time per image × retained image count × passes × configurations`, including region and masking multiplicity. Adjust batches and job decomposition using actual measurements. Cap expensive attribution on a predeclared representative evaluation subset if needed, disclose sample counts, and still evaluate every baseline; do not silently drop methods. Dataset and model variants share features wherever possible.

Track `sacct` GPU allocation, elapsed time, exit status, and the course's actual billing units. Under simple GPU-hour accounting, 2 GPUs for 8 hours consume 16 GPU-hours; confirm whether the course applies device-specific weights. Keep datasets/caches on storage that survives cloud node termination.

## Complete proposal coverage matrix

| Original scope | Current implementation | Remaining work |
|---|---|---|
| RQ1 learned regional score | Entity-deletion pilot draft | Full data builder, calibrated/semantic definition, formal training |
| RQ2 intervention consistency | Rank/locality/control code draft | Matched and natural edits; audited independent evaluation |
| RQ3 generalization | Not implemented | SigLIP and Visual Genome transfer |
| Flickr30K | Annotation archive available | Images, manifests, quality audit, full splits |
| Visual Genome | Not implemented | Data acquisition, adapter, overlap checks |
| Simple and shortcut baselines | Pilot draft | End-to-end runs, strong task-specific comparisons |
| Decomposition / Grad-ECLIP / CCI | Not implemented | Reproduction and common regional evaluation |
| GT versus generated regions | GT-only draft | Text-independent proposals, proposal recall and localization |
| Crop versus patch features | Crop-only draft | Consistent patch pooling and ablation |
| Faithfulness / recoverability | Not implemented | Independent reader, area controls, mask variants |
| Multi-seed / confidence intervals | Pilot code drafted | Actual runs, seed-level summaries and paired reporting |
| Maps / error analysis / report | Planning documents only | Visualizations, failure audit, final evidence matrix |
| Finance / CIEA adaptation | Optional, not started | Only if access/time permit |

## Handoff and first node session

Needed from the user: an existing SSH alias or login hostname and username, the allocated job ID/node or accessible terminal session, the persistent project/scratch path, and any pre-staged dataset paths. Complete MFA through the normal user login flow; no password or private key needs to be pasted into chat.

First session: inspect `nvidia-smi`, Python/CUDA compatibility, data availability and storage; stage missing models on CPU; run the small pipeline; measure throughput/peak memory; fix data validity issues; then implement and schedule the full coverage matrix. Current pilot scripts are scaffolding, not a completed implementation of that matrix.

A starter GPU script is provided in `hpc/pilot.sbatch`. Activate a suitable Python environment before submission and prepare the data first. The script intentionally fails if CUDA or the staged data are missing. It runs only the entity-deletion pilot; full experiment launchers will be added with their implementations.
