# What Does the Image Add?

Text-conditioned localization of complementary visual information — CSCI-GA 2271 course research project.

**Current priority: validate the proposal globally, audit the existing negative results, and decide whether to retain, narrow or replace the direction.** Use any available compatible GPU within the remaining authorized budget; neither L4 nor A100 is mandatory. The historical learned scorer trails inverse cosine (77.22% vs 84.50%). Independently retrained natural-caption and SigLIP results do not establish frozen transfer.

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

The full validation goal supersedes the initial local-only pilot stopping point. The CVPR extension proceeds only after its validity and usefulness gates; it is not an automatic next step. Data, downloaded weights, feature caches, and large checkpoints are not committed.
