# Beyond Mention Detection: Grounded Residual Visual Information under Text Interventions

> Superseded as the forward plan by [FOLLOWUP_PROPOSAL.md](FOLLOWUP_PROPOSAL.md), following frozen-transfer results and broader literature review. Generic coverage rewards and omission-to-crop completion have close prior art. This is retained as the earlier extension proposal; its positive-results assumptions are not established.

Research extension proposal — 26 September 2026

## Decision update after the initial HPC results

The original MLP superiority claim is unsupported: 77.22% versus 84.50% inverse-cosine deletion Acc@1; paired difference −7.28 percentage points, CI [−11.70, −3.33]. Natural-caption and SigLIP results were independently retrained, not frozen transfer. Semantic label validity and absolute-ranking tie handling remain unresolved. These findings do not justify automatically scaling the original method.

**Gate 1:** complete artifact recovery, semantic audit and frozen-checkpoint diagnostics using `hpc/NEXT_ROUND.md`. **Gate 2:** demonstrate a meaningful failure of strong direct semantic-coverage baselines on reviewed validation data. If neither task validity nor an unsolved useful problem is established, conclude the rigorous course analysis or change direction. **Gate 3:** freeze an untouched confirmation set and a fixed-reader, equal-budget evaluation before larger experiments. No claim of CVPR readiness follows from finishing the pilot.

Direction A below is conditional. Direction B may become a replacement project (budgeted acquisition of new correct facts), rather than an added module used to rescue the original story. Verify its novelty and annotation feasibility first. Direction C remains deferred. The immediately scheduled code implements Gate 1; later annotation, models and reader experiments are intentionally not represented as ready-to-run completed implementations.

**Gate outcome (HPC follow-up): closed.** Frozen deletion-checkpoint transfer did not beat inverse cosine, and only 44/100 Codex-reviewed diagnostic rows passed all semantic checks (53 invalid, 3 uncertain; zero human-confirmed rows). This document is retained as a conditional research design, but no CVPR expansion is active. Reopening it requires a repaired task, independent human labels, strong direct coverage baselines, and a new explicit decision; current compute is not authorized for that purpose.

See `RESULTS_REVIEW_AND_NEXT_STEPS.md` for stop criteria and `../GOAL.md` for hardware flexibility, productive utilization and cumulative budget. CVPR is a conference; this document targets that research standard without implying submission readiness.

## Abstract

Images contain facts that accompanying text may omit, partially express, or contradict. We propose to study grounded residual visual information at the level of region–fact pairs, and to evaluate whether a localized predictor identifies useful information that remains unavailable from text. The project will move beyond entity omission by separating visual support, textual entailment, and uncertainty, and by testing controlled text changes alongside visual evidence perturbations. The intended contribution is a rigorously audited benchmark and a method that improves budgeted information acquisition over strong coverage and grounding baselines. This is a research plan, not a claim of established novelty or a completed result.

## 1. Motivation and research gap

CIEA already learns complementary representations for retrieval. CompreCap evaluates object, attribute, and relation coverage using scene structure. CapProbe checks caption coverage using region-aligned factual questions. Consequently, spatial coverage alone is not an adequate novelty claim. The proposed research asks whether residual information can be predicted from an image and its existing text, localized reliably under semantic interventions, and used to select the next most useful region to inspect.

The central hypothesis is that explicit modeling of visual evidence and semantic coverage can outperform global embedding distance and mention detection on attributes, relations, paraphrases, and controlled visual changes. A second hypothesis is that this advantage translates into more verified new facts at a fixed region, pixel, or token budget. Either hypothesis can fail; the course pilot is a screening experiment.

## 2. Operational target

For each image, define a finite audited inventory of grounded facts F, their supporting region(s), and dependencies. Each fact receives visual-support status, text-coverage status (entailed, omitted, contradicted, ambiguous), and annotation confidence. Contradictions are evaluated separately from omissions. A person category does not exhaust facts about that person's clothing or action. Relations may be supported by multiple regions and are not forced into a single-box ontology.

Define regional residual coverage as a weighted fraction of visually supported facts not entailed by the text. Fix weights independently of test outcomes; exclude ambiguous judgments from the primary score and report their frequency. Also report unnormalized fact counts, because one missing fact out of one and ten missing facts out of ten have equal fractions but different acquisition value. This definition is relative to the inventory and annotation policy. It makes no claim to identify all scene information or a unique information-theoretic decomposition.

Distinguish two estimands:

1. Content residual: which annotated visual facts remain uncovered by text?
2. Model-relative utility: how much does a region improve an independent reader's recovery of these facts, conditional on the text?

Neither automatically proves mechanistic faithfulness to the frozen encoder. Reserve that claim for a separate behavior-linked analysis.

## 3. Benchmark construction

Start with Flickr30K Entities for entity-level supervision. Extend to densely described regions in Visual Genome and, where released and permitted, CompreCap or CapProbe as external evaluations. Audit cross-dataset overlap with image hashes/perceptual similarity before making transfer claims. Do not assume the absence of a phrase or annotation proves semantic omission.

Proposed initial annotation budget: 500–1,000 images with 3–8 selected region–fact groups per image, finalized after pilot variance and annotation-time measurements. Use two independent annotators for the primary test subset, adjudication, and agreement statistics by fact type. Document visibility, ambiguity, object size, region overlap, long-tail entities, and whether captions use synonyms or indirect references. Keep natural images and synthetic controls distinguishable.

Construct four intervention families: factual additions, semantic paraphrases, matched unrelated additions, and contradictions. Include attributes, quantifiers, hypernyms, negation, coreference, and relations. Match length and grammaticality where possible. A target addition should alter residual coverage only for the affected dependency set; a paraphrase should not. Natural alternative captions require explicit re-annotation because they often alter several facts simultaneously.

Hold out images, edit templates, and selected category–attribute compositions. Preserve an untouched natural-caption test set authored/audited independently of the training generator. Record all exclusion reasons, sources, versions, and permitted redistribution artifacts.

## 4. Method options and prioritization

### A. Factorized grounding and coverage (primary)

Predict grounded fact hypotheses from visual regions, then estimate entailment from text. Aggregate uncovered supported facts into regional scores with confidence and abstention. Train with supervised coverage and grounding signals plus directional intervention, paraphrase consistency, and dependency-aware locality losses. Contrast this with a direct MLP under matched supervision and compute.

The method is only useful if the factorization resolves demonstrated failures. Test error propagation from visual descriptions and textual entailment separately. Include an oracle-fact diagnostic to distinguish visual recognition failure from coverage failure; this is an upper-bound analysis with extra information.

### B. Conditional information acquisition (preferred application)

Given existing text and a budget, choose a region to inspect or describe next. Measure newly recovered verified facts and redundancy after adding a regional description. Compare against largest region, saliency, uncertainty, diverse selection, inverse similarity, and region-caption-plus-entailment. Hold the captioner/reader and token or pixel budget constant. Evaluate coverage gain, hallucination rate, latency, and performance as the budget varies.

If successful, this supplies a concrete reason to localize residual information, beyond producing heatmaps. It could support targeted caption completion or accessible image descriptions. User benefit would still need independent evaluation.

### C. Model-relative visual necessity (higher risk)

For independently defined questions and gold answers, estimate the change in answer recoverability when visual evidence is provided or withheld under fixed text. Cross text coverage (omitted/covered) with region visibility (present/ablated) to measure an interaction. Use matched replacements, multiple perturbations, and randomized controls. Describe this as an intervention-based model response unless identification assumptions justify a stronger causal statement. This direction is expensive and should follow the semantic benchmark.

Finance is not a core extension. It introduces temporal leakage, licensing, and confounding questions without directly resolving the main scientific gap.

## 5. Required comparisons

Strong task-matched comparisons are mandatory: region captioning followed by entailment; open-vocabulary detection plus semantic mention matching; phrase-level matching; a supervised omission classifier with the same features/data; global and regional inverse similarity. Include text-only, image-only, length, constant, and image-shuffle controls.

Add representative CLIP attribution methods for conceptual comparison, with clear score orientation and comparable region budgets. CIEA adaptations must be labeled as adaptations. Report oracle regions and text-independent automatic proposals separately; using only the paired caption to propose regions would systematically miss the target unmentioned content.

Use at least two encoder families for a generality claim and an independent reader/judge family for utility tests. These are proposed evidence requirements, not arbitrary guarantees of acceptance. Record model and prompt versions, known pretraining contamination limitations, runtime, memory, and all extra supervision.

## 6. Evaluation and statistics

Primary semantic endpoint: image-macro average precision on independent omitted-fact/region labels, with separate attribute and relation subsets. Primary intervention endpoint: target identification from per-region score changes, accompanied by signed target gap and absolute unrelated drift. Primary application endpoint: verified new facts recovered at a fixed acquisition budget.

Use image-cluster paired bootstrap confidence intervals and report effect sizes across 3–5 training seeds. Select hyperparameters and score normalization on validation only. Predefine primary comparisons and label exploratory analyses; consider multiplicity for many secondary comparisons. Determine final sample size using observed image-level paired variance and a prespecified practically meaningful effect.

Include calibration if scores are used as probabilities; abstention coverage versus error; failures by region size and overlap; paraphrase and contradiction sensitivity; out-of-domain transfer; and performance using automatically generated regions. Independently audit grader errors and judge-family sensitivity. Do not equate a larger TIG or a lower masking retrieval score with valid complementarity.

## 7. Reviewer questions that the paper must answer

- What is new relative to region-grounded caption coverage in CompreCap and CapProbe?
- Can a captioner plus entailment model already solve the task?
- Does the model need the correct image, and does it handle unseen facts rather than edit templates?
- What does an unmentioned region mean when some of its properties are stated?
- Are locality failures real errors or legitimate effects on dependent regions?
- Are human test labels independent of the training generator and automatic judge?
- Does region selection improve a useful downstream outcome at equal cost?
- Are gains robust across domains, encoders, seeds, and perturbations?
- Can another group reproduce the benchmark and analysis without access to private APIs?

## 8. Milestones and decisions

**M0, course pilot (approximately 1–2 weeks once data are available):** establish data validity, baseline difficulty, actual local throughput, and shortcut controls. Proceed only with an interpretable result; a negative finding remains a course deliverable.

**M1, definition and annotation (2–4 weeks):** finalize facts and dependency policy; annotation pilot and agreement; lock test protocol. If ambiguity dominates, narrow the task before collecting more data.

**M2, method and strong comparisons (3–5 weeks):** run factorized/direct methods, matched supervision baselines, and transfer. If a simple pipeline solves the benchmark, strengthen the substantive task or develop a benchmark/analysis paper; adding model complexity is not a solution by itself.

**M3, utility and robustness (2–4 weeks):** budgeted acquisition, independent readers, automatic regions, visual controls, and multi-seed results.

**M4, submission audit (2–3 weeks):** evidence matrix, reproducibility package, claim-to-table mapping, limitations, related-work update, and adversarial internal review. These durations are planning estimates and may overlap; no conference deadline is assumed.

Submission readiness requires a defensible contribution plus converging independent evidence, not simply completion of a checklist. Negative primary results should change the claim rather than be hidden behind favorable auxiliary metrics.

## References

- CIEA: https://aclanthology.org/2025.acl-long.1073/
- Caption/description distinction: https://aclanthology.org/2024.emnlp-main.1125/
- CompreCap: https://arxiv.org/abs/2412.08614
- CapProbe: https://arxiv.org/abs/2608.11074
- Detailed image caption benchmark: https://arxiv.org/abs/2405.19092
- CPI: https://arxiv.org/abs/2605.22651
- Visual Credit Audit (v2): https://arxiv.org/abs/2607.27069v2
- Grad-ECLIP: https://arxiv.org/abs/2502.18816
- CCI: https://arxiv.org/abs/2511.12978
- Flickr30K Entities: https://github.com/BryanPlummer/flickr30k_entities
