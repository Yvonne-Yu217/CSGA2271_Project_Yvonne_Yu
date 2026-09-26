# What Does the Image Add?

Text-conditioned localization of complementary visual information — CSCI-GA 2271 course research project.

**Current priority: validate the core original proposal within a hard 24 GPU-hour phase budget on NYU HPC, using exactly one L4 (`g2-standard-12`) for formal runs.** The repository now contains an audited Flickr30K Entities intervention experiment, three text-intervention variants, three-seed controls and ablations, spatial attribution proxies, compact quantitative results, and a separate later-stage research extension. The bounded validation result is mixed/negative: the learned scorer uses visual input but does not beat inverse crop-text cosine on the controlled benchmark, and natural-caption transfer is weak.

- [Current goal and completion criteria](GOAL.md)
- [Original proposal](proposal/What_Does_the_Image_Add_Proposal.pdf)
- [Proposal review and initial execution plan](research/REVIEW_AND_EXECUTION_PLAN.md)
- [HPC resources and full coverage matrix](hpc/RESOURCE_PLAN.md)
- [Pilot instructions and limitations](pilot/README.md)
- [CVPR extension proposal](research/CVPR_EXTENSION_PROPOSAL.md)
- [Agent log](research/AGENT_LOG.md)
- [Final evidence report](research/FINAL_REPORT.md)

The full validation goal supersedes the initial local-only pilot stopping point. The CVPR extension begins after the original proposal has been evaluated. Data, downloaded weights, feature caches, and large checkpoints are not committed.
