# Goal: Validate the proposal within a strict 24 GPU-hour budget

Updated: 2026-09-26, following the user's instruction to run the proposal first without consuming the full 300 GPU-hour course quota.

## Primary objective

Implement and validate the core claims of `proposal/What_Does_the_Image_Add_Proposal.tex` within a hard budget of 24 GPU hours. Test the three research questions at a representative, reproducible scale with the proposal's main controls, essential ablations, and failure analysis. Do not silently expand to the full course quota. A valid negative result counts as evidence; an unimplemented or unrun experiment does not.

## Completion criteria

- Flickr30K Entities: grounded intervention builder, documented official image splits, deterministic/matched/natural-caption intervention sets, data-quality audit, and leakage checks.
- Frozen CLIP scorer: ranking, locality and explicitly defined control objectives; cached features, reproducible configuration and checkpoints.
- Baselines: random/area, inverse similarity, phrase matching, text-only/image-only/shuffled-image controls, patch decomposition, Grad-ECLIP, and CCI masking importance. Add a task-matched supervised omission baseline. CIEA adaptation remains conditional, as in the original proposal.
- Evaluation: TIG, off-target stability, precisely defined target ranking and MRR, region localization evaluation, and omitted-phrase recoverability under controlled deletion. Report image-level confidence intervals and at least three training seeds.
- Essential ablations: each loss, crop versus pooled patch features, intervention construction, and at least one encoder or transfer check if the measured budget allows.
- Visual Genome: run a compact independent subset only if Flickr30K core results and feature extraction fit the 20-hour budget; otherwise record it as explicitly unverified rather than consuming quota blindly.
- Final report: claim-to-evidence table, quantitative tables, qualitative maps, error taxonomy, compute ledger, reproducible commands, and Agent Log. Explicitly identify unsupported claims.
- Push reproducible code, configurations, documentation and compact results to the existing GitHub repository; keep datasets, caches, credentials and bulky checkpoints out of Git.

Optional finance and licensed WSJ experiments remain conditional on data access and remaining resources, consistent with the original proposal. The CVPR extension is a separate later research phase. An access-dependent exception must be disclosed and agreed with the user; it cannot silently count as a completed experiment.

## Resource policy and order

Use account `ds_ga_1006_001-2026fa`, but treat 24 GPU hours as the project hard cap for this phase. Use exactly one L4 (`g2-standard-12`) for formal runs; do not request A100 or multi-GPU jobs. Stop jobs before the cap and record `sacct` usage.

Order: CPU data staging → L4 smoke test → Flickr30K core scorer and controls → essential ablations → one expensive baseline or compact transfer check → report. Reuse cached features and stop if the measured remaining budget cannot support a result. Do not spend the allocation on a Cartesian product of configurations.

HPC utilization is an execution constraint: do not leave allocated GPU or CPU nodes at persistently low utilization. Stage inputs before allocation, keep the device fed, emit frequent progress/utilization records, and promptly diagnose, restart, or cancel stalled jobs before the scheduler reclaims them. If a primary GPU run stalls, switch to a predeclared productive fallback such as feature caching, a baseline, an ablation, or evaluation; do not use synthetic burn loops merely to inflate utilization. If no useful task is ready, release the allocation. Cache and checkpoint reusable work so recovery does not repeat expensive computation.

## Current state

- The bounded Flickr30K Entities validation is implemented and has run end to end on one L4. The audited subset contains 350 images and 1,165 grounded intervention pairs with fixed 200/50/100 image splits.
- CLIP deterministic, length-matched, and natural-caption interventions completed with three training seeds, image-level bootstrap confidence intervals, loss ablations, and random/area/inverse-similarity/phrase/text-only/image-only/shuffled-image/supervised controls.
- Algorithm-adapted pooled-patch, region masking, CCI clustering, and Grad-ECLIP attribution completed. Absolute localization and omitted-phrase recoverability checks completed. SigLIP encoder transfer completed for all three interventions.
- The main superiority claim is not supported at this scale: on deterministic deletion, rank+locality+control Acc@1 is 0.7722 versus 0.8450 for inverse cosine. The negative result and failure taxonomy are retained rather than hidden.
- Visual Genome and a distinct automatic-region transfer benchmark remain explicitly unverified. They are not required for this bounded core result and were not added after the central hypothesis failed.
- Reproducible code, compact metrics, qualitative figures, compute ledger, report, and Agent Log are prepared in the repository. Raw data, model caches, and bulky predictions remain excluded from Git.

## Exact text for the conversation goal

/goal 第一目标：在不超过 24 GPU hours 的硬预算内，完成当前 proposal 的核心验证。正式运行使用一张 L4（`g2-standard-12`），切换不增加预算。覆盖 Flickr30K Entities、三项研究问题、主要基线与负控、核心文本干预、必要消融、至少三次 seed（预算允许时）和置信区间；Visual Genome、SigLIP、自动区域与昂贵 attribution 只在剩余预算足够时执行，否则明确标记未验证。保存可复现实验、失败分析、Agent Log 和最终证据报告并推送 repo。完成原 proposal 的预算内验证后，再按独立 CVPR proposal 推进投稿级研究。一定要注意保持 GPU 占用，要不我们研究的数据都难以下载下来，必须时刻监测并保持占用。

The repository goal and active conversation goal use the objective above.
