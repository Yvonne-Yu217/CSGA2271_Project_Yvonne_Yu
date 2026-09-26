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
- **E1 is not formally evaluated.** Recognition-oracle utility and strong-baseline
  utility still need adjudicated action outcomes. Independent-model sensitivity
  screens and the direct full-image completion baseline are decision diagnostics,
  not E1 evidence. Built-in rules remain prohibited from passing the gate.
- **E2 is not evaluated.** Enriched, paraphrase, and saturated contexts still
  need independent construction/review. STOP must be measured using the frozen
  deployable baseline, never the recognition oracle.
- E3 selector training, E4 independent confirmation, E5 publication expansion,
  and downstream finance work remain gated on E0–E2.

## Provisional automated direction screen

The 100-image/500-caption screen in `research/PROVISIONAL_DIRECTION_SCREEN.md`
began with same-family judgments, then added independent SmolVLM crop support and
DeBERTa novelty filters. The latter preserve a 23.2-point strict core-target
oracle-minus-montage gap. However, the proposal-required caption-conditioned
direct full-image baseline changes the decision: at a 50,176-pixel cap it reaches
48.6% provisional strict core-target success versus 30.8% for the fixed-candidate
oracle. Direct completion minus oracle is 17.8 points [9.2, 26.2]. Automated
semantic typing is not E1 evidence, but this is strong evidence against investing
in the current candidate/action design before a small stopping audit.

## Immediate next decision

Do not start E3 or a broad 100-image annotation campaign on the current action
inventory. First complete the prepared 80-context, 160-row-per-reviewer stopping
audit comparing low-resolution direct completion with crop-oracle claims. If
independent humans reverse the automated ordering, resume canonical fact review
and formal E0. Otherwise retain the negative diagnostic, replace the action/task
design or change topic, and use new development data for the next candidate.
The subsequent automated claim-conditioned crop-verification screen already
failed its separate recall/false-support rule and must not be trained. It does
not change the value of the small human stopping audit for closing the original
direction, but it is not a fallback research route.

## Resource/session note

The L4 interactive allocation is preserved for server-managed reclamation; it
is not manually canceled or released. Productive GPU work is monitored, cached,
and never replaced with synthetic utilization. Exact Slurm elapsed time is a
snapshot while job 1979 remains live and therefore is not a final ledger value.
