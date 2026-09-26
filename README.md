# What Does the Image Add?

Text-conditioned localization of complementary visual information — CSCI-GA 2271 course research project.

**Milestone order: CV validation → course-project ready → CVPR ready → downstream application.** Finance follows core CV development: first grounded event understanding, then conditional market prediction. The current authoritative proposal is `research/FOLLOWUP_PROPOSAL.md`; downstream planning does not change the immediate E0–E2 priority.

**Current priority: establish a positive result with a defensible contribution through a task/method pivot.** Actual frozen transfer completed without learned superiority. The proposed next task predicts which observation adds new grounded facts given an existing description, especially attributes and relations of already mentioned entities. First measure strong-baseline and oracle performance, then decide whether to train. Research design is no longer constrained by the old pilot-hour envelope; use compatible HPC resources and maintain accounting.

- [Latest results, audit corrections and literature/repo review](research/DIRECTION_DECISION_AFTER_FROZEN_RESULTS.md)
- [Proposed next task and course-to-publication experiment design](research/FOLLOWUP_PROPOSAL.md)

- [Result review and direction decision](research/RESULTS_REVIEW_AND_NEXT_STEPS.md)
- [Completed frozen-follow-up code and CPU → GPU → CPU commands](hpc/NEXT_ROUND.md)
- [Current goal and completion criteria](GOAL.md)
- [Historical pilot proposal (PDF)](proposal/What_Does_the_Image_Add_Proposal.pdf)
- [Proposal review and initial execution plan](research/REVIEW_AND_EXECUTION_PLAN.md)
- [HPC resources and full coverage matrix](hpc/RESOURCE_PLAN.md)
- [Pilot instructions and limitations](pilot/README.md)
- [CVPR extension proposal](research/CVPR_EXTENSION_PROPOSAL.md)
- [Agent log](research/AGENT_LOG.md)
- [Final evidence report](research/FINAL_REPORT.md)
- [100-row Codex semantic review—not human annotation](research/semantic_audit_codex.csv)

The full validation goal supersedes the initial local-only pilot stopping point. The CVPR extension gate is currently closed. Data, downloaded weights, feature caches, raw predictions, and large checkpoints are not committed.
