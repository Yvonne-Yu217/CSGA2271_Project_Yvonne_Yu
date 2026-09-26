# Goal: Validate complementary visual information through better methods

Updated 2026-09-26 after pulling HPC commit `2f96409` and the user's latest correction. This is the authoritative execution goal; it supersedes earlier instructions to abandon the research question after a single method family fails.

## Objective and priority

**CV validation → course-project ready → CVPR ready → downstream application.**

Keep the research question: **given an image and an existing description, extract additional correct, useful visual facts, with evidence tied to the right entities.** Missing attributes and relations of mentioned entities remain the primary diagnostic slice. Region selection, segmentation, global views and fact selection are replaceable implementation choices.

Continue developing and testing methods toward an independently confirmed positive result and a defensible contribution. A failed method is a reason to diagnose and redesign, not automatically abandon the question. Positive results are a completion requirement, not something the agent can promise or manufacture.

Authoritative proposal: [FOLLOWUP_PROPOSAL.md](research/FOLLOWUP_PROPOSAL.md).
Next experiment and handoff plan: [METHOD_REDESIGN_PLAN.md](research/METHOD_REDESIGN_PLAN.md).
Device/execution policy: [RESOURCE_PLAN.md](hpc/RESOURCE_PLAN.md).

## What the latest evidence establishes

- The original CLIP MLP did not reliably beat inverse cosine. Preserve its results.
- The next acquisition implementation used **fixed 3×3 tiles, horizontal/vertical halves and a full-image action**, not learned semantic segmentation. Its crop observer was caption-independent, while the stronger direct full-image baseline received the caption and a mentioned-entity completion prompt.
- Automated strict-target success was 48.6% for low-resolution direct completion versus 30.8% for the fixed-candidate recognition oracle, difference +17.8 percentage points [9.2, 26.2]. This rejects investment in that unchanged pipeline; it does not isolate geometry, establish segmentation as the cause, or disprove complementary-information learning.
- Judges were unstable: same-family Qwen accepted all 1,281 non-empty observations; SmolVLM accepted 393. Neither is human ground truth. Crop-verification “false support” is disagreement against full-image model judgments, not an adjudicated factual false-positive rate.
- E0 human adjudication is unfinished. E1/E2 have diagnostic screens, not formal passes. Never relabel proxy results as validated facts.
- The focus-ambiguity detour is archived, including the paired protocol. It is not the active main line. Its official test corpus was repurposed by that protocol; do not describe it as a pristine test for future development.

See the unchanged historical screens in `research/PROVISIONAL_DIRECTION_SCREEN.md`, `research/NEXT_DIRECTION_AFTER_STOPPING_SCREEN.md` and `research/FOCUS_AMBIGUITY_PREFLIGHT.md`. Their old topic-change recommendations are superseded by this goal; their results remain evidence.

## Next method and completion gates

1. **R0 / E0: establish valid evidence and diagnose the failure.** Reuse old outputs only for diagnosis. Complete the blinded stopping audit; also audit a random sample, because disagreement-enriched samples do not estimate population error. Stage new image-disjoint development data and independent entity/fact/context labels. Missing human evidence remains an explicit dependency, never filled by an AI pretending to be a human.
2. **R1 / E1: isolate the implementation defects.** Compare old grids with automatic grounded entity boxes, padded boxes and relation-union views. Cross geometry with caption-aware versus caption-independent observation and crop-only versus global-plus-region context. Use identical frozen observer, prompts where applicable, output limits and measured total costs. GT regions are diagnostic upper bounds only.
3. **R2 / E1–E2: measure residual value over strong full-image completion.** Preserve full-image completion as a first-class action/baseline. Test whether another observation adds verified facts beyond it, against an equal-call full-image refinement baseline. Validate same-image description changes, paraphrase stability, same-instance binding and stopping. Do not train a crop selector if the redesigned actions have no useful residual headroom.
4. **R3 / E3: train only the mechanism supported by R1/R2.** Candidate method: instance-linked known-fact state plus direct marginal-value prediction for global, entity, relation and STOP actions. Compare a strong prompted planner, all-observations semantic ranking with its full cost, no-context/no-binding/no-paired-supervision ablations and three seeds. If selection is unnecessary, test the global fact-selection family described in the proposal while preserving the objective.
5. **R4 / E4: course-project confirmation.** Freeze the selected design and strongest comparator before one untouched confirmation set. Require a practically meaningful gain, paired image-level uncertainty, human verification, error control and reproducible code. A prompt-only or detector-only improvement can establish engineering progress, but does not by itself meet the novelty requirement.
6. **R5 / E5: CVPR development.** After R4, establish prior-art distinction, cross-domain/backbone/reader transfer, automatic proposals, cost–quality curves and mechanistic ablations. Multi-step policies are conditional on additional headroom. Readiness is a research standard, not a guarantee of acceptance.
7. **D1 / D2: downstream application.** After the CV package, test finance event understanding with full article and image caption available at the observation time. Market prediction follows only if valid data and D1 support it; it is not required to rescue CV novelty. Follow the proposal's time, duplicate and leakage controls.

## Mandatory autonomous failure-to-redesign loop

When a method fails, the task remains unfinished. Within the authorized compute balance, the agent must:

1. Save the run manifest, all primary/secondary results, uncertainty, costs and representative failures.
2. Classify the bottleneck: labels/judge, candidate coverage, entity binding, context loss, visual resolution, generation, action prediction, cost or lack of novelty. Separate observations from hypotheses.
3. Revisit relevant primary papers and official code. Propose a material change targeted at the diagnosed bottleneck; do not merely retune the same model indefinitely.
4. Write the next hypothesis, discriminating controls, fixed primary metric, continuation rule, sample split and finite job queue **before** running it. Implement/stage that round before GPU submission.
5. Run the development experiment, record its decision, and repeat with a new design when warranted. After two failed variants addressing the same bottleneck, change method family rather than repeat parameter sweeps.
6. Continue toward independent positive evidence and novelty. If labels, access or authorized compute block execution, report the exact dependency and continue independent preparation; do not silently mark the objective complete or consume unauthorized quota.

All previously inspected samples remain development data. Retire a failed confirmation set from confirmation use; obtain a fresh holdout for a revised method. Never switch metrics, cherry-pick seeds or repeatedly inspect a holdout until it becomes positive. If multiple well-controlled method families leave no support for the scientific hypothesis, present that evidence and discuss revising the question rather than claiming success.

Fresh holdouts alone do not control repeated-confirmation false positives. Before the first formal confirmation, register a project-level attempt limit and multiplicity/alpha-spending plan, shared across method versions. An initial plan is at most two confirmatory attempts with two-sided alpha 0.025 each (97.5% intervals for the primary gain), totaling at most 0.05; adjust further if multiple primary comparisons are introduced. Report all attempts. Continued experimentation after that limit is development, not another unadjusted opportunity to declare success. Development screens may retain descriptive 95% intervals.

## GPU utilization is an execution priority

**Remember this at every handoff: keep allocated GPUs doing necessary, prepared work; detect and address low utilization promptly. Do not manually release or cancel allocations.**

- During an active HPC run, sample GPU utilization, used/available memory and progress every **5 seconds**. Record throughput, CPU/I/O waits, run ID and job ID; maintain a monitor independent of the experiment process. A stale heartbeat is an issue even if utilization is high.
- In steady GPU-ready stages, target **70–90% or higher when useful and feasible**. Investigate utilization below 30% for two minutes or stalled progress immediately. Also inspect rolling utilization and end-to-end useful outputs/hour; a high peak or occupied VRAM is not sustained productive work.
- Prepare code, pinned dependencies, model weights, data, cached inputs and resume commands before the allocation's compute stage. Maintain a finite manifest-backed queue of approved, scientifically necessary inference, baselines and ablations. Queue completion, model loading, CPU annotation and failures can cause brief dips; do not promise physically constant 100% utilization.
- Use batching, resolution/length buckets, CPU workers, pinned-memory prefetch and cached region preparation. While one GPU job runs, subagents may prepare independent code/data/literature/analysis. One owner alone submits jobs, assigns output directories and accounts for aggregate resource use.
- Prefer one well-fed GPU process per device initially. Add concurrent inference workers only after measured memory headroom and throughput justify them; agents must not independently load several models onto the same GPU. Two GPUs require enough ready work to feed both.
- On a stalled workload, preserve artifacts, diagnose, and move to the next valid ready task. Do not keep rerunning completed experiments, leak held-out data, or generate burn loads to fill the GPU.
- **Never manually release, cancel, `scancel`, or restart existing CPU/GPU allocations.** Preserve the working session, checkpoints, caches and environments; the server handles reclamation. Ending a completed experiment process or unloading its model within the retained allocation is allowed for the next prepared task.
- When no valid GPU-ready work exists, log the dependency, advance CPU/preparation work and preserve the allocation for server-managed reclamation. Do not invent work or silently cross a scientific gate.
- No live HPC connection or utilization trace was inspected in this local documentation turn. These are execution requirements, not a claim that a remote GPU is currently busy.

## Available-GPU execution and accounting

**Use any currently available suitable GPU, including L4 or A100, and finish the required experiments.** Start with one device, record its exact type, and adapt model size, resolution, batching and staged loading to measured capacity and throughput. Previous L4 acquisition runs used about 7.4–7.7 GiB peak torch allocation; a separate Qwen ambiguity workload reached 10.79 GiB. These are workload-specific measurements, not a bound for the new global-plus-region setup or another GPU type.

Profile actual processed visual tokens, batch size, peak allocated/reserved/device memory, valid outputs/second and total stage time on every device used. If a configuration does not fit, adapt it or use another available GPU and disclose any resulting scientific limitation. Do not cancel an allocation merely to change GPU type.

The research design is not constrained by historical pilot-hour envelopes. Actual spending still requires current authorized balance, cumulative `sacct` reconciliation and allocation-aware accounting; **300 course GPU-hours is not authorization to spend them all**. Old 20/24-hour figures and the 2.0333-hour snapshot are historical, not a verified current balance. No new compute is submitted by this documentation update.

## Suggested conversation goal

`/goal 保持“给定已有文本，提取图片能补充的正确且有用事实”这一主线，按 CV 验证 → course-project ready → CVPR ready → downstream application 推进。把失败归因到具体方法和证据，先公平比较区域生成、caption 条件、全局上下文和事实评估，再学习相对强全图基线的新增收益。若方法失败，必须保存结果、分析错误、查阅论文与官方 repo、自主提出并预先记录新实验设计，在授权资源内继续迭代直到获得独立确认的正向效果与明确新意；不能反复刷测试集或包装假阳性。可使用任何当前可用且适合的 GPU（包括 L4 或 A100），从单卡开始，按实际硬件调整模型、分辨率、训练与 batch，并把必要实验跑完。GPU 有效占用是每次执行和交接的重点：每 5 秒监控，及时诊断低利用率，提前准备代码数据和有限必要任务队列，必要时安排 subagent 并发准备，统一提交记账。不得随意取消、释放或重启 CPU/GPU 分配，保留缓存检查点与会话，由服务器管理回收。`

This file and its suggested command update the repository goal; no conversation-level goal API rewrite is claimed.

## Latest redesign execution (2026-09-26)

R1 and R2 now have automated development results; see
`research/R1_R3_REDESIGN_RESULTS.md`. Caption-conditioned fixed-grid actions
show strict same-image caption-dependent oracle headroom (+10.2 points over the
best static action [7.6, 13.0], and +41.8 points over direct full-image
completion [34.8, 49.0]). This is a positive mechanism diagnostic, not human
evidence. Existing prompted planners do not capture it, and the first frozen
SigLIP interaction selector failed its uncertainty rule. Do not tune its proxy
test. Continue R1 geometry/binding isolation with automatic grounded entity,
padded and relation-union views; complete human E0 before treating selector
training as formal E3 or opening a confirmation attempt.

Automatic grounding is now implemented. Pooled grounded actions retain 87.0%
strict automated oracle success versus 90.4% for grids and use fewer actions;
unpadded entity boxes retain positive headroom, while padding and relation-union
families fail. Explicit Grounding-DINO phrase binding also fails to improve the
proxy selector. Preserve these results and do not tune the inspected split.
Before choosing another selector family, independently test the Qwen strict
semantic-type labels and prepare human review of R2 action-value disagreements.

That independent SmolVLM sensitivity is now complete and preserves the
caption-dependent oracle effect (grid +6.8 points; grounded +3.4 points). The
required equal-call full-image refinement control is also complete: two
high-resolution full-image calls cover 59.2% strictly, still 31.2 points below
the fixed-grid oracle and 27.8 points below the pooled grounded oracle. These
are automated development results, not human evidence. The next gate remains
human E0 and blinded review preparation; do not tune another selector on the
already-inspected proxy split.

R2 paraphrase stability is also complete on development data. Oracle
availability is stable on the 435/500 paraphrases that pass bidirectional NLI,
but exact successful-action sets have only 0.72--0.74 mean Jaccard across
wordings. Treat selector targets as noisy/set-valued, and include the 65
automatically disputed paraphrases in human review rather than excluding them
from the evidence record.

The latest R2 blinded packet is staged at
`acquisition/data/e0-r2-blind-review-100`: 100 image-clustered random rows for
population error estimation and 50 rows each for grid judge disagreement, grid
consensus positives, grounded consensus positives, and paraphrase action flips.
Both reviewer sheets have 300 independently shuffled rows and 600 verified
assets. Its audit status is `awaiting_reviews`; blanks are not labels.
