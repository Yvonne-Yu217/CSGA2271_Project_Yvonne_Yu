# VQ-FocusAmbiguity preflight

This directory contains a reproducible, test-label-locked preflight for the
[official VQ-FocusAmbiguity dataset](https://vizwiz.org/tasks-and-datasets/focus-ambiguity-in-visual-questions/).
Downloaded images, masks, model features, and predictions are ignored by Git.

## Data audit

Place the official archives and `train_gt.json`, `val_gt.json`, and
`test_gt.json` under `focus_ambiguity/data/source`, and extract images and masks
under `focus_ambiguity/data/extracted`. Then run:

```bash
.venv/bin/python focus_ambiguity/audit_data.py \
  --source focus_ambiguity/data/source \
  --extracted focus_ambiguity/data/extracted \
  --output focus_ambiguity/results/data_audit.json
```

The audit checks global IDs, mask IDs/counts, binary nonempty masks, actual
image/mask dimensions, source hashes, split overlap, and exact cross-split
image-question duplicates. The JSON file containing a record is authoritative
for its split: all 5,474 records have the literal internal field `set: train`,
so the field is unusable for splitting. Forty image filenames cross official
files, although no exact normalized image-question pair does. Any internal
resampling must therefore group by image filename.

The number of focus masks must not be used as the ambiguity label. Some
unambiguous questions legitimately have multiple answer/focus groundings, and
two ambiguous records have one complete focus mask.

## Frozen zero-shot baseline

The following evaluates only the 140 public train+val examples and leaves the
5,334-record test file untouched by inference or model selection:

```bash
.venv/bin/python focus_ambiguity/run_frozen_baseline.py \
  --source focus_ambiguity/data/source \
  --images focus_ambiguity/data/extracted/images \
  --output focus_ambiguity/results/smolvlm-train-val \
  --batch-size 4
```

It uses pinned SmolVLM-Instruct and compares question-only with image+question
classification. Predictions, raw generations, prompt fingerprint, revision,
runtime, and GPU peak are cached. A/B labels are used because the first format
probe showed that a conventional prompt caused the VLM to answer the embedded
question instead of classifying its focus.

Use `--prompt-variant swapped` with a different output directory to run the
predeclared label-order sensitivity check. A reliable semantic classifier should
not change materially when the meanings of A and B are exchanged; the current
SmolVLM and Qwen results do, so their prompted scores are diagnostics rather
than accepted baselines.

## Fixed linear learnability probe

This probe extracts pinned SigLIP features and fits one deterministic dual
ridge classifier with `lambda=1`; it performs no hyperparameter search:

```bash
.venv/bin/python focus_ambiguity/run_frozen_siglip_probe.py \
  --source focus_ambiguity/data/source \
  --images focus_ambiguity/data/extracted/images \
  --output focus_ambiguity/results/siglip-fixed-ridge \
  --batch-size 16
```

Question-only, image-only, and concatenated image+question probes train on the
70 official train rows and are evaluated once on the 70 validation rows. Test
labels are never used for fitting, thresholding, or selecting a modality.
The same script accepts `--model` and `--revision` for a pinned cross-encoder
replication; the preflight also records a frozen CLIP run.

The current decision and exact results are in
`research/FOCUS_AMBIGUITY_PREFLIGHT.md`. These scripts establish a feasibility
screen, not a new method or publication result.
