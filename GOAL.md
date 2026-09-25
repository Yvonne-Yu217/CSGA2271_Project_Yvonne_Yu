# Goal: Validate the proposal within a strict 20 GPU-hour budget

Updated: 2026-09-26, following the user's instruction to run the proposal first without consuming the full 300 GPU-hour course quota.

## Primary objective

Implement and validate the core claims of `proposal/What_Does_the_Image_Add_Proposal.tex` within an initial hard budget of 20 GPU hours. Test the three research questions at a representative, reproducible scale with the proposal's main controls, essential ablations, and failure analysis. Do not silently expand to the full 300 GPU-hour quota. A valid negative result counts as evidence; an unimplemented or unrun experiment does not.

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

Use account `ds_ga_1006_001-2026fa`, but treat 20 GPU hours as the project hard cap for this phase. The remaining course quota is not part of the plan. Use one A100 (`c12m85-a100-1`) for the formal proposal run because it gives more throughput and memory headroom for Grad-ECLIP, CCI masking, and repeated experiments. A single L4 (`g2-standard-12`) is the fallback if A100 is unavailable; changing hardware does not increase the 20 GPU-hour cap. Stop jobs before the cap and record `sacct` usage.

Order: CPU data staging → A100 smoke test → Flickr30K core scorer and controls → essential ablations → one expensive baseline or compact transfer check → report. Reuse cached features and stop if the measured remaining budget cannot support a result. Do not spend the allocation on a Cartesian product of configurations.

## Current state

- Proposal review and separate CVPR extension proposal written.
- Local pilot code drafted; parser/syntax checks completed; end-to-end experiments and training have NOT run.
- Official annotation archive downloaded and CRC checked. Image sample acquisition was stopped at the user's request before completion.
- Local experimental processes stopped. Awaiting HPC connection details and node allocation before GPU experiments.
- Full experiment modules beyond the entity-deletion pilot still require implementation; `pilot/run.py` is not a full proposal runner.

## Exact text for the conversation goal

/goal 第一目标：在不超过 20 GPU hours 的硬预算内，完成当前 proposal 的核心验证，而不是消耗课程提供的全部 300 GPU hours。正式运行使用一张 A100（`c12m85-a100-1`），因为需要为 Grad-ECLIP、CCI masking 和重复实验保留吞吐与显存余量；A100 不可用时才使用一张 L4（`g2-standard-12`），切换不增加预算。覆盖 Flickr30K Entities、三项研究问题、主要基线与负控、核心文本干预、必要消融、至少三次 seed（预算允许时）和置信区间；Visual Genome、SigLIP、自动区域与昂贵 attribution 只在剩余预算足够时执行，否则明确标记未验证。保存可复现实验、失败分析、Agent Log 和最终证据报告并推送 repo。完成原 proposal 的预算内验证后，再按独立 CVPR proposal 推进投稿级研究。

The repository goal above is updated. The conversation goal is currently paused; the exposed goal API supports creation and status changes but not replacement of an unfinished objective. Its text has not yet been changed through that API. Use the exact `/goal` command above in the conversation to update it through the user command surface.
