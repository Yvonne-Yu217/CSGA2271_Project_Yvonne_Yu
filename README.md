# What Does the Image Add?

Text-conditioned localization of complementary visual information — CSCI-GA 2271 course research project.

**Decision reached: retain this phase as a rigorous negative course comparison and repair or replace the task before more learned-method work.** Frozen deletion-checkpoint transfer did not beat inverse cosine, and a 100-row Codex visual diagnostic found only 44 valid rows (53 invalid, 3 uncertain; no human-confirmed labels). Conditional confirmation, Visual Genome and CVPR expansion remain stopped at the gate.

- [Result review and direction decision](research/RESULTS_REVIEW_AND_NEXT_STEPS.md)
- [Next-round code and CPU → GPU → CPU commands](hpc/NEXT_ROUND.md)
- [Current goal and completion criteria](GOAL.md)
- [Revised proposal](proposal/What_Does_the_Image_Add_Proposal.pdf)
- [Proposal review and initial execution plan](research/REVIEW_AND_EXECUTION_PLAN.md)
- [HPC resources and full coverage matrix](hpc/RESOURCE_PLAN.md)
- [Pilot instructions and limitations](pilot/README.md)
- [CVPR extension proposal](research/CVPR_EXTENSION_PROPOSAL.md)
- [Agent log](research/AGENT_LOG.md)
- [Final evidence report](research/FINAL_REPORT.md)
- [100-row Codex semantic review—not human annotation](research/semantic_audit_codex.csv)

The full validation goal supersedes the initial local-only pilot stopping point. The CVPR extension gate is currently closed. Data, downloaded weights, feature caches, raw predictions, and large checkpoints are not committed.
