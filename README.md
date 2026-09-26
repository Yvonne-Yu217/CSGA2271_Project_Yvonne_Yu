# What Does the Image Add?

Text-conditioned localization of complementary visual information — CSCI-GA 2271 course research project.

**Milestone order: CV validation → course-project ready → CVPR ready → downstream application.** Finance follows core CV development: first grounded event understanding, then conditional market prediction. The current authoritative proposal is `research/FOLLOWUP_PROPOSAL.md`; downstream planning does not change the immediate R0–R2 priority (repair and validate E0–E2).

**Current priority: retain the complementary-information question and redesign the failed method.** The latest fixed-grid/caption-independent crop pipeline lost to caption-conditioned full-image completion; that comparison does not isolate segmentation or invalidate the research objective. Next isolate grounding, text conditioning and global context, then test residual facts beyond strong full-image baselines. Follow the diagnosis → redesign → development → independent-confirmation loop until positive evidence and novelty meet the goal. Use any suitable available GPU, with continuous productive-utilization monitoring and no manual allocation release.

- [Current proposal and course-to-publication experiment design](research/FOLLOWUP_PROPOSAL.md)
- [Latest failure audit, redesign experiments and HPC handoff](research/METHOD_REDESIGN_PLAN.md)
- [GPU resource and utilization plan](hpc/RESOURCE_PLAN.md)
- [Historical frozen-transfer literature/repo review](research/DIRECTION_DECISION_AFTER_FROZEN_RESULTS.md)
- [Archived focus-ambiguity preflight](research/FOCUS_AMBIGUITY_PREFLIGHT.md)
- [Archived focus-ambiguity reproduction commands](focus_ambiguity/README.md)

- [Result review and direction decision](research/RESULTS_REVIEW_AND_NEXT_STEPS.md)
- [Completed frozen-follow-up code and CPU → GPU → CPU commands](hpc/NEXT_ROUND.md)
- [Current goal and completion criteria](GOAL.md)
- [Historical pilot proposal (PDF)](proposal/What_Does_the_Image_Add_Proposal.pdf)
- [Proposal review and initial execution plan](research/REVIEW_AND_EXECUTION_PLAN.md)
- [Pilot instructions and limitations](pilot/README.md)
- [CVPR extension proposal](research/CVPR_EXTENSION_PROPOSAL.md)
- [Agent log](research/AGENT_LOG.md)
- [Final evidence report](research/FINAL_REPORT.md)
- [100-row Codex semantic review—not human annotation](research/semantic_audit_codex.csv)

R0–R2 method diagnostics are the immediate priority; the proposed redesign is not yet implemented. Formal human validation and the CVPR extension gate remain incomplete. Data, downloaded weights, feature caches, raw predictions, and large checkpoints are not committed.
