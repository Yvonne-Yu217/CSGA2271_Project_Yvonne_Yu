# NYU HPC resource and execution plan

## Request now: 24 GPU-hour hard cap

Based on the partitions and quota provided by the user:

| Resource | Initial request |
|---|---|
| Account | `ds_ga_1006_001-2026fa` |
| Required partition | `g2-standard-12` |
| Required GPU | 1 × NVIDIA L4 24GB |
| CPU | 8 cores initially; up to 12 if useful |
| Host memory | 64 GB |
| Initial wall time | 4 hours, after CPU-side data/model staging |
| Persistent/scratch capacity | 200 GB preferred; increase if retaining full raw datasets and many caches |
| Software | Python 3.11/3.12, CUDA-compatible PyTorch, Transformers; Git and outbound data access or a staged transfer |

For formal proposal runs, use one L4 on `g2-standard-12`, as explicitly required by the current goal. Control memory with batching and cached features. Do not request A100, two GPUs, or distributed training. The total phase cap is 24 allocated GPU hours.

Use CPU partition `n2c48m24` for downloads, extraction, annotation validation, and cached-feature small-scorer training. Do not hold a GPU while waiting for data. Exact account permissions, allowed wall times, GPU GRES spelling, and charge multipliers are not yet verified on the cluster.

Example allocation, using standard Slurm syntax; confirm site settings first:

```sh
sinfo -p g2-standard-12 -o '%P %l %G %m %c'
salloc --account=ds_ga_1006_001-2026fa \
  --partition=g2-standard-12 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=04:00:00
srun --pty bash -l
```

Sources for generic flags: [Slurm salloc](https://slurm.schedmd.com/salloc.html), [Slurm sbatch](https://slurm.schedmd.com/sbatch.html). Partition/account values come from the user, not these generic documents.

## Strict planning envelope, not measured runtime

| Work package | GPU-hour ceiling to plan against |
|---|---:|
| L4 smoke test and throughput measurement | 1 |
| Flickr30K feature extraction and core scorer | 8 |
| Essential controls, losses, and seeds | 7 |
| One compact transfer or expensive baseline check | 4 |
| Final rerun and figures | 4 |
| **Hard phase cap** | **24** |

These are scheduling caps, not evidence that every run will finish inside them. After the fixed 100-image smoke test, estimate `time per image × retained image count × passes × configurations`, including region and masking multiplicity. If the estimate exceeds 24 GPU hours, reduce the predeclared representative subset and mark omitted proposal components as unverified; do not silently request more quota. Dataset and model variants share features wherever possible.

Track `sacct` GPU allocation, elapsed time, exit status, and the course's actual billing units. Under simple GPU-hour accounting, 2 GPUs for 8 hours consume 16 GPU-hours; confirm whether the course applies device-specific weights. Keep datasets/caches on storage that survives cloud node termination.

## Executed proposal coverage matrix

| Original scope | Executed evidence | Status / limitation |
|---|---|---|
| RQ1 learned regional score | Three-seed CLIP scorer versus all declared controls | Primary superiority claim not supported |
| RQ2 intervention consistency | Deterministic, length-matched, and natural same-image captions | Controlled edits transfer; natural transfer weak |
| RQ3 generalization | SigLIP across all three interventions | Encoder tested; Visual Genome unverified |
| Flickr30K | 350 audited images, 1,165 grounded pairs, fixed 200/50/100 splits | Complete for bounded validation |
| Visual Genome | Not run | Explicitly unverified |
| Simple and shortcut baselines | Random/area, inverse similarity, phrase matching, text/image/shuffled controls, supervised omission | Complete |
| Decomposition / Grad-ECLIP / CCI | Pooled patch plus algorithm-adapted Grad-ECLIP and CCI | Authors' repositories not executed |
| GT versus generated regions | CCI text-independent patch clusters with K sensitivity; evaluation mapped to GT boxes | Distinct proposal generator/recall study unverified |
| Crop versus patch features | Crop and pooled-patch ablation | Complete |
| Faithfulness / recoverability | Absolute localization and masked omitted-phrase similarity across interventions | Complete; learned recovery intervals include zero |
| Multi-seed / confidence intervals | Three fixed seeds and image-level bootstrap intervals | Complete |
| Maps / error analysis / report | Qualitative maps, error taxonomy, compact metrics, final evidence report | Complete |
| Finance / CIEA adaptation | Optional, not started | Only if access/time permit |

## Reproduction and subsequent node sessions

The completed run uses the pinned environment in `requirements.txt` and the launchers in this directory. Stage data with `pilot/prepare.py`, verify it with `pilot/audit.py`, then submit `hpc/full_l4.sbatch`; the transfer, attribution, and faithfulness scripts are separate so failures can resume from cached artifacts.

Every GPU job records progress and utilization. If preprocessing or model loading creates a gap, queue another declared experiment rather than synthetic work; release the allocation when the evidence bundle is finished.
