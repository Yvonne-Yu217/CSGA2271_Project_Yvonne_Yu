# Agent log

## 2026-09-26 — Direction review after the frozen follow-up

- Fast-forward pulled `7716d0f` to `5f24bd4`; reviewed the actual frozen transfer tables and automatic semantic audit with a read-only audit subagent. No new HPC experiment was launched.
- Frozen ensemble deletion Acc@1 remains below inverse cosine by 5.42 percentage points, paired CI [−10.42, −0.67]. Matched/natural transfer gives no reliable superiority. Small reviewed subsets cannot establish a positive claim.
- Corrected a material interpretation risk: 44/100 passing all checks is not a dataset-wide semantic-error estimate. In the recorded audit, all 23 deletion failures are grammar-only; natural controls fail in 23/33 rows. Some automatic grammaticality judgments are inconsistent. Human adjudication remains needed.
- Identified objective mismatch: within-region caption-difference losses do not directly supervise absolute region choice or actual new-fact acquisition. More compute alone does not resolve that mismatch.
- Conducted targeted primary-source and official-repository research, assisted by a literature subagent. Checked CompreCap, CapRL/CapRL++, CCCaption, ClaimDiff-RL, AdaptVision, SC-Captioner, CaptionQA, CapProbe and the 2025 high-resolution omission/crop/refinement predecessor. Generic coverage rewards and omitted-object crop completion are already covered by prior work.
- Read official code entry points for CompreCap evaluation, AdaptVision training and CapRL Prism evaluation. Cloned CompreCap into an isolated temporary directory at commit `b70aae69f3e3516caf93eb455faa3e8f96e54668` for inspection; did not install or run third-party training. Some repositories have API dependencies or incomplete releases, documented in the review.
- Added a detailed direction decision and separate follow-up proposal: context-dependent visual acquisition targeting missing attributes/relations of mentioned entities, actual action-value supervision, strong full-image/planner/coverage baselines, oracle headroom, independent confirmation and conditional publication expansion. Novelty and positive gains are hypotheses, not established results.
- Updated repository goal to the user's positive-results and novelty requirement. Old pilot-hour envelopes no longer constrain the research design. Actual execution still requires profiling/accounting and preserves the no-manual-cancellation session policy. Historical proposal PDF and numeric evidence remain unchanged.
- This turn implements planning/documentation only; the new E0–E2 pipeline is explicitly not yet implemented. Checked documentation diff and local link targets. No training/code regression suite was needed for these document-only changes.

## 2026-09-26 — Frozen transfer, semantic review, and decision gate

- Latest user override after the follow-up: never manually release, cancel, or `scancel` CPU/GPU allocations because doing so interrupts the task session; the server handles reclamation. This supersedes earlier project-memory advice to release idle nodes. Productive work remains preferred and synthetic load remains prohibited.
- Job 1975 is therefore left running for server-managed reclamation. Its elapsed time is recorded only as a generation-time ledger snapshot; no artificial or duplicate GPU task was started after the experimental decision gate closed.

- Recovered the persistent formal artifacts and ran `pilot/next_round.py preflight` against the deletion source. It verified split separation, staged images, checkpoint dimensions, hashes, cached encoder identity, and emitted a 100-row stratified semantic-audit sample. The legacy training-time encoder revision remains unprovable and is disclosed.
- Ran `pilot/visibility_audit.py`: among 342 unique historical regions, 190 are fully visible in the center crop, 141 partially visible, and 11 invisible; 19 require the historical nearest-patch fallback. This invalidates an unqualified crop-versus-patch comparison.
- Ran one bounded frozen inference pass on L4 job 1972. All 925 pairs were encoded jointly; the same three deletion-trained checkpoints produced predictions for 337 deletion, 337 matched, and 251 natural-caption pairs without retraining. Five-second monitoring captured a 91% utilization peak. Job 1972 was canceled after GPU work and CPU summary completed; final elapsed time was 683 seconds.
- Frozen prediction-ensemble delta Acc@1 was 0.7908/0.7317/0.5601 for deletion/matched/natural, versus inverse cosine 0.8450/0.7575/0.5816. On the full deletion set, the paired ensemble-minus-inverse interval remained below zero. Frozen transfer therefore does not reverse the primary negative result.
- Rendered all 100 semantic-audit rows with target/control boxes and original/edited captions, then visually reviewed every page. This was a Codex model review, explicitly **not** a human annotation: 44 rows passed all checks, 53 failed, and 3 were uncertain. Common failures were broken deletion grammar, invalid `unspecified` generalizations, target synonyms/implications retained in natural captions, and changed non-target facts.
- Updated the summary schema to distinguish reviewed-valid rows from human-confirmed rows. Human-confirmed rows remain zero; downstream reporting must not relabel the model review as human ground truth.
- Decision gate outcome: finish the course project as a rigorous negative comparison and task-validity analysis. Do not run learned variants, untouched-set confirmation, Visual Genome, or CVPR expansion on the current labels. Repair or replace the task before additional method development.
- The interactive notebook repeatedly received a fresh L4 allocation after cancellation. Each allocation with no valid GPU-ready task was released rather than filled with duplicate or synthetic computation; this session overhead is recorded in the final Slurm ledger.
- The compute node had no `pdflatex`. A downloaded glibc Tectonic binary was incompatible with the node's older libc; the static musl build succeeded. The revised proposal compiled to four pages and all four rendered pages were visually checked for clipping and layout failures.

## 2026-09-26 — Post-HPC evidence review and prepared follow-up

- Pulled HPC result commit `6d67ef4`. Reviewed compact metrics and implementations with an independent audit subagent. No HPC connection, new training, or GPU allocation occurred in this session.
- Corrected interpretation: three-seed metric means are not prediction ensembles; natural/matched/SigLIP scorers were independently retrained. Natural-caption locality assumptions, unreviewed semantics, scale sensitivity, fixed-index tie bias, spatial crop visibility, cache provenance and incomplete accounting remain material limitations. Historical metric values were preserved.
- Revised GOAL.md, project memory, resources, proposal and extension gates. The first objective is global evidence-based proposal validation, with explicit permission to conclude a negative course analysis or change direction. Removed exclusive L4/A100 restrictions. Recorded five-second monitoring, productive utilization targets, CPU/GPU staging, subagent coordination, and release of unneeded allocations.
- Imported HPC history states a revised 24-hour cap; earlier local authorization states 20. The next handoff must reconcile this and final usage, using the lower cap if unresolved. The 1.1267-hour historical snapshot is not a final balance. No extra allocation was authorized or consumed here.
- Added `pilot/next_round.py` and `hpc/next_round.sbatch`: CPU preflight and 100-pair semantic audit template; one shared feature extraction across interventions; inference using the original three frozen deletion checkpoints; CPU tie-aware summaries, paired bootstrap, scale normalization, candidate-count strata, audit-valid/common subsets, and distinct seed mean versus prediction ensemble.
- Preflight checks checkpoint/model dimensions, hashes images/source artifacts/current encoder snapshot/implementation, and fails on changed inputs. Legacy metadata cannot prove the original training-time encoder revision; this limitation remains explicit.
- Added CPU `pilot/visibility_audit.py` for center-crop visibility and nearest-patch fallback counts. Corrected future ledger generation to multiply elapsed time by allocated GPU count without double-counting typed GRES entries.
- Validation: Python compilation and both CLI help commands passed; Slurm shell syntax passed; subagent checks covered ties, normalization, bootstrap and in-memory summary/audit gating; root checks covered full/partial/invisible crop geometry and 0/1/2-GPU accounting. `git diff --check` passed. Actual dataset preflight, CUDA extraction and HPC throughput are not executed locally because the required HPC artifacts are unavailable here.
- Recompiled the revised proposal to three pages and visually inspected all pages after rendering. No clipping or unresolved citations. A missing local Courier font was resolved by using Computer Modern typewriter for the file-path text.
- Only the bounded validity/frozen-transfer round is runnable now. Conditional new models, independent reader studies and publication-scale confirmation are future gated work, not claimed implemented experiments. GOAL.md includes proposed `/goal` text; the existing conversation objective cannot be rewritten by the exposed goal API.

## 2026-09-26 — Review, planning, and local pilot

- Read the complete proposal LaTeX source. Preserved the original `.tex` and `.pdf`, which already had user changes.
- Checked primary literature records and relevant full-text sections. Added CompreCap and CapProbe to the novelty assessment. The review is targeted, not exhaustive.
- Confirmed local CLIP loading and PyTorch MPS availability. No external inference or paid compute was used.
- Initial sequential download of the 27.9 MiB annotation archive timed out after 120 seconds. A remote fsspec annotation-index attempt also stalled. Replaced this with bounded concurrent HTTP range downloads, response-length/range checks, and ZIP CRC validation.
- Implemented a restricted entity-omission pipeline; explicitly did not treat deleted-phrase labels as complete semantic coverage labels.
- Checked caption token spans, one-based to half-open box conversion, overlap exclusion, and repeated-entity rejection with constructed examples. Python compilation checks passed.
- Designed constant, length, image-only, text-only, and shuffled-image controls to expose trivial solutions. A positive TIG by itself is not accepted as direction validation.
- Prepared separate course review/execution plan and publication extension proposal. Actual experiment outcomes will be appended after execution.

## HPC handoff update

- User changed the primary objective to complete proposal-wide validation using the provided 300 GPU-hour HPC allocation.
- Stopped the active local preparation process. Annotation archive CRC verification succeeded; image staging and all training remain incomplete.
- Added `GOAL.md`, an explicit full-scope coverage matrix, resource budget, and starter Slurm script. The pilot is clearly labeled incomplete and narrower than the full proposal.
- Added CUDA selection and configurable input/output paths to the pilot runner. No cluster job has been submitted and CUDA execution is unverified.

## Budget update

- User set the current allocation to 20 GPU hours and explicitly requested not to consume the full 300-hour course quota.
- Finalized one A100 (`c12m85-a100-1`) as the preferred formal node because the proposal includes attribution and repeated masking; one L4 (`g2-standard-12`) is the fallback. The total phase cap remains 20 GPU hours.

## 2026-09-26 — Executed bounded validation

- The user revised the active cap to 24 GPU hours and required formal runs on exactly one L4 (`g2-standard-12`). Updated the repository goal, resource plan, project memory, and Slurm launchers accordingly. Earlier A100 runs are retained only as exploratory/runtime evidence; the final evidence bundle is the L4 run.
- Fixed annotation staging when `pilot/data` does not yet exist. Downloaded the official Flickr30K Entities annotation archive, verified ZIP CRC, staged 350 images, and verified all 350 image SHA256 hashes.
- Found and fixed 19 apparent post-deletion phrase-presence warnings caused by repeated lexical mentions. Rebuilt the dataset; the final audit contains 1,165 pairs, no split-image leakage, no box/hash failures, and no warnings.
- Added deterministic deletion, length-matched generalization, and natural same-image caption interventions. Natural candidates exclude the exact target entity ID and phrase but can still contain implicit/synonymous mentions; this is recorded as label noise.
- Moved small-scorer fitting onto the selected CUDA device after a smoke test caught and fixed one CPU/CUDA target mismatch. Added manifest snapshots and content fingerprints to prevent stale feature reuse.
- Ran the formal CLIP bundle on one L4 with three fixed seeds, image-level bootstrap confidence intervals, loss ablations, text-only/image-only/shuffled-image controls, inverse similarity, n-gram similarity, area/length/constant controls, and a supervised omission baseline.
- Added pooled-patch, region-masking, and gradient-times-activation spatial baselines on the held-out test split. These are task-adapted proxies, not official Grad-ECLIP or CCI reproductions, and are labeled accordingly.
- The primary superiority claim was not supported: inverse crop-text cosine exceeded the learned rank+locality+control scorer on deterministic deletion. Shuffling image features reduced learned performance, showing visual dependence without demonstrating advantage over the simple baseline. Natural-caption transfer was substantially weaker.
- Added 10-second GPU/CPU/RSS utilization logging and a project-memory rule to queue productive fallback work, cancel/restart stalled jobs, and never use synthetic burn loops merely to inflate utilization.
- Generated compact metrics, a claim-to-evidence report, and qualitative success/failure maps. Visual Genome and automatic-region transfer remain unverified; they must not be represented as completed.
- The first SigLIP attempt failed before GPU computation because `sentencepiece`, then `protobuf`, were absent. Added both pinned dependencies, verified `SiglipProcessor` plus 768-dimensional image/text features on CPU, and reran successfully on one L4 across all three seeds.
- Added absolute region selection and omitted-phrase recoverability. The first 10-second utilization trace missed a roughly 13-second burst, so repeated the same valid evaluation with 2-second sampling; results were unchanged apart from runtime metadata and the trace captured a 73% GPU-utilization sample.
- Replaced the initial generic gradient proxy with a Grad-ECLIP algorithm adaptation using the published final-attention CLS gradient, value tokens, and CLS–patch key cosine weights. Added a CCI adaptation using deterministic cosine clustering of patch tokens and CLS-to-cluster attention masking in the last vision transformer block. These are Hugging Face CLIP adaptations, not execution of the authors' repositories.
- Used the live L4 allocation for three additional, predeclared evidence packages rather than synthetic utilization: SigLIP matched/natural transfer, CLIP attribution under matched/natural interventions, and matched/natural localization/recoverability. All completed successfully and wrote isolated artifacts.
- Added a CCI automatic patch-cluster-count sensitivity check at K=3/8/12. Deterministic-deletion Acc@1 was 0.6958/0.7558/0.7675, compared with 0.7204 at the original K=5 run. This tests internal cluster sensitivity but is not a separate proposal-generator/recall benchmark.
- SigLIP matched rank+locality+control Acc@1 was 0.5528 (three-seed ensemble computed in the final report) and natural-caption performance remained near chance; the exact image-level intervals and all methods are in `pilot/results/final_metrics.json`.
- The first root-level unittest invocation lacked `PYTHONPATH=pilot` and failed to import `prepare`; rerunning with the documented module path passed all four tests. Python compilation, data audit, and `git diff --check` also passed.
