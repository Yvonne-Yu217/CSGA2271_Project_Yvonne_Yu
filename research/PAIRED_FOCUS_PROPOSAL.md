# Same-image paired focus-ambiguity falsification proposal

> Archived side-branch protocol, not the current execution plan. No paired inference result is recorded by this document. Resume the complementary-information method redesign in [METHOD_REDESIGN_PLAN.md](METHOD_REDESIGN_PLAN.md); retain this protocol and its split provenance for history.

Date: 2026-09-26. Predeclared before any paired model inference.

## Motivation

The initial VQ-FocusAmbiguity preflight found that a fixed CLIP image-only probe
outperformed its image+question probe. The official dataset mixes four sources,
including an all-unambiguous MSRA subset, so aggregate accuracy can reward
source recognition without question-specific focus reasoning.

The official raw test JSON contains 2,206 PACO images with exactly two records:
one ambiguous question and one unambiguous question. This supports a stricter
same-image task. A valid model must distinguish two questions while the pixels
and source remain fixed. Image-only predictions cannot answer both records
correctly or rank one above the other.

This is a new internal benchmark protocol, not official test performance. The
official test corpus is repurposed as raw labeled data, so results must never be
compared with official leaderboard numbers.

## Frozen protocol

- Exclude every image filename occurring in the official 70-row train or
  70-row validation JSON.
- Retain only PACO images with exactly one ambiguous and one unambiguous record.
- Assign whole images using `sha256("paired-focus-v1:" + file_name) mod 10`:
  buckets 0–5 train, 6–7 validation, and 8–9 locked holdout.
- The frozen split contains 1,283 train pairs (2,566 records), 484 validation
  pairs (968 records), and 439 locked-holdout pairs (878 records). Its manifest
  SHA-256 is
  `215752e67a55eded41941a61aea91c0972838ab1adecc40c368aeb3b205338c9`.
- Freeze the manifest hash and counts before model inference.
- Do not load or encode holdout images during development.

The first probe uses pinned frozen SigLIP and CLIP features. It fits one
deterministic `lambda=1` ridge probe with no tuning for question-only,
image-only, concatenated image/question, and elementwise image–question
interaction features. Primary metrics are same-image pair ranking and pair
accuracy; record accuracy is secondary. Bootstrap units are image pairs.

## Continue and stop rules

Continue only if the interaction probe exceeds question-only pair-ranking by at
least 5 points and the paired-bootstrap 95% interval for the difference is
strictly above zero, independently for both SigLIP and CLIP. Image-only must
remain at zero pair accuracy and 50% tie-aware ranking, confirming the control.

Stop if either encoder fails. Do not tune the split, ridge coefficient,
threshold, features, or prompts after observing validation. Do not open the
holdout merely because validation is negative.

## Novelty boundary

The original ambiguity-recognition/localization tasks and the 2026
disambiguation-sufficiency reformulation are existing work. The candidate
contribution is specifically a same-image paired protocol that removes
image/source shortcuts and requires question–image interaction. Before any
method development, a further literature review must confirm that the official
papers and follow-ups do not already report this paired protocol or an
equivalent image-controlled evaluation.
