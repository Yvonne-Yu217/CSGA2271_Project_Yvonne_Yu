# E0–E2 implementation and evidence status

Date: 2026-09-26. This is a status record, not a positive-results claim.

## Completed engineering work

- Staged 100 new Flickr30K train images disjoint by source ID from the historical
  350-image pilot, with 500 natural captions, 1,500 fixed automatic actions
  (14 image observations plus STOP per image), image SHA-256 hashes, and
  perceptual-duplicate checks. Source entity phrases/boxes remain review seeds,
  not facts or gold labels.
- Froze Qwen2.5-VL-3B-Instruct at snapshot
  `66285546d2b821cf421d4f5eb2576359d3770cd3` and cached all 1,500 actions.
  Generation for 1,400 non-STOP observations took 455.72 seconds on one NVIDIA
  L4 and peaked at 7,721.54 MiB allocated torch memory. The automatic format
  audit found no empty outputs, while 119 returned `NO_VISIBLE_FACT`, 8 hit the
  96-token cap, and roughly 200 exceeded the requested 30 words. Outputs can
  hallucinate on small crops and require blind claim review.
- Cached a CLIP ViT-B/16 diagnostic similarity baseline for 500 contexts and
  1,400 non-STOP candidates (7,000 same-image scores; 21.89 seconds). This old
  cache recorded revision `main`; new runs pin commit
  `57c216476eefef5ab752ec549e440a49ae4ae5f3` and verify cache hashes/counts.
- Added a frozen Qwen full-image prompted planner baseline. All 500 contexts
  completed in 116.11 seconds on one L4 (peak 7,671.47 MiB torch allocation);
  deterministic parsing recovered all 500 choices, including 85 STOP choices.
  Its cache is model-visible only and cannot establish performance until E0
  labels exist.
- Generated review-only enriched, paraphrase, and saturated caption drafts for
  all 100 images in 180.42 seconds on one L4 (peak 7,519.10 MiB torch
  allocation). All 100 outputs parsed into the three requested fields and none
  copied enriched text verbatim into paraphrase text. These drafts are neither
  verified captions nor context labels and must be independently reviewed.
- Added strict public/gold separation, exact Cartesian atomic-label coverage,
  exactly two votes plus explicit adjudication, instance/image consistency,
  STOP semantics, deterministic action derivation, duplicate-group bootstrap,
  a two-stage blind review export, and counterexample tests.

## Gates not completed

- **E0 is not passed.** There are zero completed pairs of independent human
  reviews and no adjudicated exhaustive fact inventory. Existing generated
  review packets are templates only; fact/instance review must precede context
  and observation-claim labeling.
- **E1 is not evaluated.** Recognition-oracle utility and strong-baseline
  utility need adjudicated action outcomes. Built-in area, proposal-score,
  center, random, and STOP rules are diagnostics and are prohibited from
  passing the evidence gate. Full inference cost must be included.
- **E2 is not evaluated.** Enriched, paraphrase, and saturated contexts still
  need independent construction/review. STOP must be measured using the frozen
  deployable baseline, never the recognition oracle.
- E3 selector training, E4 independent confirmation, E5 publication expansion,
  and downstream finance work remain gated on E0–E2.

## Provisional automated direction screen

The non-independent 100-image/500-caption screen in
`research/PROVISIONAL_DIRECTION_SCREEN.md` found a 16.4-point binary any-new-fact
oracle gap over the montage planner and a much larger provisional gap for the
intended mentioned-entity-detail target. This is sufficient to prioritize human
validation, not sufficient to pass E0/E1/E2 or claim a positive result.
The strict instance-linkage prompt preserves a 27.8-point oracle-minus-montage
gap but reduces oracle availability to 34.8%; broad/strict target labels have
only 16.65% Jaccard. This prompt sensitivity is now part of the review design.

## Immediate next decision

Complete two-person fact/instance annotation on at least 100 images, adjudicate
unknowns, freeze canonical IDs, then independently label context coverage,
candidate visibility/sufficiency, and observer claims. Only then compute E1/E2
with image/duplicate-cluster macro intervals. If oracle headroom or genuine
caption dependence is absent, change direction before selector training.

## Resource/session note

The L4 interactive allocation is preserved for server-managed reclamation; it
is not manually canceled or released. Productive GPU work is monitored, cached,
and never replaced with synthetic utilization. Exact Slurm elapsed time is a
snapshot while job 1979 remains live and therefore is not a final ledger value.
