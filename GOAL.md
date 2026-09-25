# Goal: Validate the full proposal before pursuing the CVPR extension

Updated: 2026-09-26, following the user's instruction to prioritize global proposal validation and move experiments to NYU HPC.

## Primary objective

Implement and validate the full core experimental scope of `proposal/What_Does_the_Image_Add_Proposal.tex`, using the available DS-GA 1006 Cloud Bursting HPC allocation. Test all three research questions with reproducible quantitative results, negative controls, ablations, transfer experiments, and failure analysis. A small local pilot is a prerequisite only and does not complete this objective. Do not require positive results: a valid negative result counts as evidence; an unimplemented or unrun experiment does not.

## Completion criteria

- Flickr30K Entities: grounded intervention builder, documented official image splits, deterministic/matched/natural-caption intervention sets, data-quality audit, and leakage checks.
- Frozen CLIP scorer: ranking, locality and explicitly defined control objectives; cached features, reproducible configuration and checkpoints.
- Baselines: random/area, inverse similarity, phrase matching, text-only/image-only/shuffled-image controls, patch decomposition, Grad-ECLIP, and CCI masking importance. Add a task-matched supervised omission baseline. CIEA adaptation remains conditional, as in the original proposal.
- Evaluation: TIG, off-target stability, precisely defined target ranking and MRR, region localization evaluation, and omitted-phrase recoverability under controlled deletion. Report image-level confidence intervals and at least three training seeds.
- Ablations: each loss, crop versus pooled patch features, CLIP versus SigLIP, ground-truth versus automatically generated regions, intervention construction, and training/transfer settings.
- Visual Genome: independent dense-domain evaluation and documented overlap checks. Complete RQ3 beyond Flickr30K.
- Final report: claim-to-evidence table, quantitative tables, qualitative maps, error taxonomy, compute ledger, reproducible commands, and Agent Log. Explicitly identify unsupported claims.
- Push reproducible code, configurations, documentation and compact results to the existing GitHub repository; keep datasets, caches, credentials and bulky checkpoints out of Git.

Optional finance and licensed WSJ experiments remain conditional on data access and remaining resources, consistent with the original proposal. The CVPR extension is a separate later research phase. An access-dependent exception must be disclosed and agreed with the user; it cannot silently count as a completed experiment.

## Resource policy and order

Use account `ds_ga_1006_001-2026fa` with the user-reported 300 GPU-hour allocation. Prefer one A100 40GB, first allocation 8 hours after CPU-side data staging. Measure actual throughput before scaling. Keep a 60 GPU-hour reserve and an initial 240 GPU-hour planning envelope; allocations are not measured runtime guarantees. Track actual cluster accounting and any course-specific charging multipliers.

Order: data/environment → small end-to-end check → full Flickr30K scorer and controls → attribution baselines → ablations → Visual Genome transfer/automatic regions → faithfulness and final report. Reuse cached features. Do not spend all compute on a Cartesian product of configurations.

## Current state

- Proposal review and separate CVPR extension proposal written.
- Local pilot code drafted; parser/syntax checks completed; end-to-end experiments and training have NOT run.
- Official annotation archive downloaded and CRC checked. Image sample acquisition was stopped at the user's request before completion.
- Local experimental processes stopped. Awaiting HPC connection details and node allocation before GPU experiments.
- Full experiment modules beyond the entity-deletion pilot still require implementation; `pilot/run.py` is not a full proposal runner.

## Exact text for the conversation goal

/goal 第一目标：在 NYU DS-GA 1006 HPC 的 300 GPU-hour 配额内，完整实现并验证当前 proposal 的全部核心内容，而非仅完成小规模 pilot。覆盖 Flickr30K Entities、Visual Genome、三项研究问题、主要基线与负控、三类文本干预、loss/特征/编码器/区域生成/跨域消融、faithfulness、多 seed 和置信区间；保存可复现实验、失败分析、Agent Log 和最终证据报告并推送 repo。先在单张 A100 40GB 上测量吞吐再扩大实验，保留算力余量；如受数据访问或资源阻塞，明确列出未验证项，不宣称完成。金融和 CIEA 扩展按原 proposal 的 optional 范围处理。完成原 proposal 的全局验证后，再按独立 CVPR proposal 推进投稿级研究。

The repository goal above is updated. The conversation goal is currently paused; the exposed goal API supports creation and status changes but not replacement of an unfinished objective. Its text has not yet been changed through that API. Use the exact `/goal` command above in the conversation to update it through the user command surface.
