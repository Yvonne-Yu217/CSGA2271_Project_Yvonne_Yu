# Agent log

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
