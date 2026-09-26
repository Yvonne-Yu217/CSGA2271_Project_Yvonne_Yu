# Goal: Establish a positive result with a defensible research contribution

Updated 2026-09-26 after pulling HPC commit `5f24bd4` and the user's new direction request. The user requires positive results and novelty for the course project as well as any publication extension. A negative-results report alone no longer satisfies the intended outcome.

## Current decision

Stop expanding the original CLIP-crop MLP and phrase-deletion objective as the main direction. Frozen ensemble intervention Acc@1 is 79.08/73.17/56.01% for deletion/matched/natural, versus inverse cosine 84.50/75.75/58.16%. No reliable learned superiority has emerged. Keep historical artifacts as evidence.

Do not interpret 44/100 passing automatic audit as a population semantic-error rate: 23 deletion failures were grammar-only, natural-caption controls genuinely changed, and some automatic judgments are inconsistent. New data needs adjudicated semantic labels.

The next candidate is **learning which visual observation adds the most grounded facts given an existing description**, particularly missing attributes/relations of already mentioned entities. This is a proposed pivot, not an established novel or successful method. Generic omission detection, crop-and-caption refinement, QA rewards and adaptive cropping have close predecessors.

Authoritative design: `research/FOLLOWUP_PROPOSAL.md`. Evidence and literature/repo review: `research/DIRECTION_DECISION_AFTER_FROZEN_RESULTS.md`. The existing proposal PDF and results document the completed phase; do not overwrite history with unrun positive claims.

## Work and completion gates

**Priority order: CV validation → course-project ready → CVPR ready → downstream application.** Finance is a conditional downstream extension, not the main objective or a substitute for a positive CV result. Current work is CV validation (E0–E2). Prepare downstream data only when it does not delay that work.

1. **E0: valid task and data.** Prepare new image-disjoint development data, fluent caption states, independently checked object/attribute/relation facts and automatic candidates. Check instance identity, visibility, changed facts and annotation agreement. Preserve unknowns.
2. **E1: measure available improvement.** Freeze the observer, enumerate candidate outcomes, and compare attainable same-budget utility against full-image caption completion, a strong prompted planner, omission rules, semantic coverage and simple selection. Count all inference costs. If the oracle has no useful gap, replace the direction before training.
3. **E2: test caption dependence.** Show that useful actions change when known facts change, remain stable under paraphrase, and lose value after redundant acquisition. Include natural captions and saturated-caption STOP cases.
4. **E3: learn actual action value.** Only after E1/E2, train direct utility ranking/regression and paired-caption supervision. Compare equal supervision, fixed observer and three seeds; avoid indefinite architecture/hyperparameter sweeps.
5. **E4: independent positive confirmation.** Freeze model, metrics and strongest comparator; evaluate untouched images with human fact checks and image-level paired intervals. A meaningful gain over a competitive baseline is required. Negative outcomes change the next decision, not the stored results.
6. **E5: publication development.** Only after positive confirmation and a clear prior-art distinction, test cross-reader/domain/backbone transfer, automatic regions, sequential observations, stop calibration and full quality–cost curves. A checklist cannot guarantee publication readiness.
7. **D1: downstream event understanding, after the CV publication package.** Apply the validated method to financial news images. Condition on the full article and image caption available at the observation time. Compare text-only, full-image, generic image descriptions and selected complementary facts, plus shuffled-image controls. Require grounded event-understanding gains; finance alone does not establish method novelty.
8. **D2: conditional market-outcome study.** Only if D1 succeeds and time-aligned data is valid, test incremental prediction of a preregistered market outcome with chronological evaluation and leakage controls. Better visual facts do not imply better returns prediction. A null market result must be retained and does not invalidate independently confirmed CV results.

Course-project readiness requires E0–E4, a reproducible implementation, human checks, strong baselines and mechanism ablations. CVPR readiness additionally requires E5 and a defensible distinction from prior work; it is a research target, not an acceptance guarantee. Downstream work follows these milestones and must not postpone the core CV confirmation to chase a financial correlation.

Sample sizes and suggested thresholds are in the proposal; they are planning choices, not forecasts. Next implementation is E0–E2 infrastructure. The old frozen-checkpoint scripts do not implement the new task.

## Resources and execution

- Design the study around the scientific question. The user removed the prior limited-resource assumption for planning. Old 20/24-hour and four-hour diagnostic envelopes describe the completed pilot, not the new research design.
- Before actual submission, profile model/resolution, check nodes/current allocation and maintain a cumulative ledger. The last committed 2.0333 GPU-hour figure includes a running allocation and is not final usage. No new job was submitted during this planning update.
- Use any suitable available GPU, including L4 or A100. Neither is mandatory. Match model size, precision, batch size and parallelism to measured memory/throughput; stage independent inference work for available nodes.
- Prepare code, models/data and CPU preflight before allocation. Keep third-party reproduction environments separate from the original pilot. Save revisions, prompt hashes, sample IDs, cost and resume artifacts.

## Productive utilization and session continuity

Persistently low utilization slows progress and must be diagnosed. Sample memory/utilization and progress every five seconds. During steady GPU-ready stages, target 70–90% where feasible; investigate <30% for two minutes or stalled throughput. These are internal diagnostics, not scheduler guarantees. Improve batching, data workers, I/O and caching. Move small statistical jobs to CPU.

Maintain a finite queue of necessary work. Subagents may prepare independent code/data/analysis; one owner controls GPU submissions, shared output paths and total cost. Never repeat experiments or generate synthetic load to inflate utilization. Under the latest HPC instruction, **do not manually cancel or release existing CPU/GPU allocations**; preserve the session and let the server handle reclamation.

Prepare and push runnable stage code, dependencies, inputs and resume commands before GPU execution so implementation gaps do not strand allocated hardware. If a run stalls, diagnose it and switch only to a ready, scientifically necessary task within the current gate. Preserve allocations, checkpoints, caches and working environments; do not arbitrarily delete resources or restart the allocation to address low utilization.

## Suggested conversation goal

`/goal 按 CV 核心验证 → course-project ready → CVPR ready → downstream application 的顺序推进，主次分明。首先验证给定已有描述时，能否学习选择最能补充正确视觉事实的观察，重点是已提及实体的未覆盖属性/关系。先完成可靠数据、强 baseline、oracle 和上下文必要性验证，再训练实际收益选择器并做独立测试，以可信正向效果、人工核验和明确新意达到课程标准；随后完成跨域/跨模型、机制与质量成本证据，推进 CVPR 级研究。此后开展 finance 事件理解；只有数据与结果支持时才研究市场预测，不能用金融相关性替代 CV 方法验证。研究设计不受旧试验小时数限制，使用任何合适的可用 GPU，提前完成代码与数据准备，保持有效 GPU 占用并及时诊断低利用率，必要时用 subagent 准备独立任务；统一提交、记账、缓存和检查点，不随意取消或释放 CPU/GPU 分配，不随意删除资源，由服务器管理回收并保持会话连续。若核心方向无提升空间或无实质新意就换题，绝不强行包装正结果。`

This updates the repository objective; it does not claim the conversation-level goal was rewritten through an API.

## Current implementation state (2026-09-26)

The E0–E2 infrastructure, 100-image staging, frozen observer/planners, independent
SmolVLM visual-support screen and independent DeBERTa novelty screen now exist;
see `research/E0_E2_PROGRESS.md` and `research/PROVISIONAL_DIRECTION_SCREEN.md`.
E0 still has no completed pair of independent human annotations or adjudication
and therefore has **not passed**. Generated saturated contexts failed the natural
E2 screen, so E2 has not passed either.

The required caption-conditioned direct full-image completion baseline changes
the provisional direction decision. At a 50,176-pixel cap it retained 55.8%
automated any-new-fact success and 48.6% strict mentioned-entity-detail success.
The fixed-candidate strict core-target oracle reached only 30.8%; direct completion
exceeded it by 17.8 points with image-bootstrap interval [9.2, 26.2]. These are
automated sensitivity labels rather than human evidence, but the current action
inventory cannot beat a goal-aligned strong baseline even with oracle selection.
Do **not** train E3 on this candidate design. Use only a small blinded stopping
audit to test whether the ordering is a judge artifact; otherwise replace the
task/action design or change topic, as required by the goal, rather than expanding
annotation or presenting the earlier oracle gap as a positive result.

An adjacent claim-conditioned crop-verification pivot was also screened and
failed its predeclared stop rule. Proper zoom views recovered 80.30% of
full-image-supported claims but falsely supported 57.56% of rejected/partial
claims, versus a 20% ceiling. Local tiles reduced false support to 14.38% only by
collapsing recall to 42.76%. Generic zoom/verification also has direct recent
prior art. Do not train this pivot. Preserve both negative screens and move to a
genuinely different CV task with externally verifiable labels; do not skip to
finance or reweight these proxies to manufacture a positive result.

A first externally labeled topic preflight now exists for VQ-FocusAmbiguity;
see `research/FOCUS_AMBIGUITY_PREFLIGHT.md`. Its official images and 15,361
masks pass structural validation, but the metadata require care: every internal
`set` field says `train`, 40 image filenames cross official JSON files, and 11
width/height records are swapped while image and mask pixels agree. Use the JSON
filename as the split and group any internal resampling by image.

This topic has **failed the direction gate in its current form**. A pinned zero-shot SmolVLM
image+question classifier reached only 51.67% balanced accuracy and 3.33%
ambiguous recall on the 140 public train+validation rows. Swapping the meanings
of A/B changed its ambiguity predictions from 2 to 23; the same change moved
Qwen2.5-VL-3B from 66 to 3, with only 55% semantic agreement across Qwen prompt
orders. Treat prompted generative classification as invalidated by label-order
bias. A fixed, untuned
SigLIP image+question ridge probe trained on 70 rows reached 60.83% balanced
accuracy on 70 validation rows, but its 95% bootstrap interval [49.48, 72.02]
includes chance and performance varies sharply by source. A fixed CLIP probe is
more concerning: image-only balanced accuracy is 62.50% [51.19, 73.56], higher
than image+question at 57.50% [45.67, 69.05], indicating source/image shortcuts
rather than demonstrated question-specific focus reasoning. Order-symmetrized
next-token logits also fail: Qwen reaches 52.92% balanced accuracy
[47.89, 58.21] and SmolVLM 50.83% [50.00, 52.73]. The official test set
remains locked. Recent ICCV 2025 and CVPRW 2026 work already covers ambiguity
recognition, focus localization, sufficiency-oriented evaluation, and a
two-stage baseline. Do not scale training or submit test predictions. Archive
this preflight and change to another externally verifiable CV topic; reopening
it requires a genuinely new contribution and source-balanced data, not tuning
the 140 public development rows.
