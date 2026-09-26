# Local entity-omission pilot

This bounded validation tests entity-phrase deletion on actual Flickr30K images and official Entities annotations. It does **not** establish semantic complementarity, human annotation agreement, or publication readiness. SigLIP transfer, attribution adaptations, patch-cluster sensitivity, and recoverability are included; Visual Genome, an independent proposal-generator benchmark, and downstream studies remain future work.

## Reproduce

Run from the repository root with Python 3.12 and the packages recorded in `results/feature_runtime.json` (once extraction completes). Additional dependencies are NumPy, Pillow, requests, fsspec, and aiohttp. On the original Mac, the existing local Hugging Face cache supplies `openai/clip-vit-base-patch16`; the runner defaults to local-only loading. Stage weights before GPU allocation, or explicitly pass `--allow-model-download` on a network-enabled staging session.

```sh
python3 -u pilot/fetch_annotations.py
python3 -u pilot/prepare.py --counts 200,50,100
python3 -u pilot/audit.py
python3 -u pilot/run.py
python3 -u pilot/baselines.py --device cuda
python3 -u pilot/faithfulness.py --device cuda --run-dir RUN_DIR
python3 -u pilot/make_report.py --formal-root FORMAL_RUN_DIR
```

No paid services are used. Public image samples are read from the `nlphuji/flickr30k` archive; annotations and split lists come from Bryan Plummer's official repository. This mirror is not the official image request channel. Respect original Flickr image rights and research/education restrictions. Images and model caches are excluded from Git.

For the formal L4 bundle, submit `sbatch hpc/full_l4.sbatch`. It runs deterministic deletion, length-matched generalization, natural same-image caption pairs, and the held-out spatial proxies serially on one L4 while recording utilization every ten seconds. `pilot/run.py --minus-field {tminus,tmatched,tnatural}` selects an intervention and `--model` selects a compatible Hugging Face vision-language encoder.

## Protocol

- Seed 2271 shuffles each official image split independently. Select the first eligible 200 train, 50 validation, and 100 test images.
- A usable caption has at least two single-box, single-mention entities with disjoint boxes at least 16 pixels in each dimension. The target and all retained controls are disjoint; controls may overlap one another.
- Remove one annotated phrase as the target intervention; remove a different retained entity phrase for the unrelated-edit control. Keep up to four pairs per image. These deterministic edits can be ungrammatical and can retain indirect references. This is an explicit validity limitation.
- Use frozen normalized CLIP crop/text features, batch 16, MPS where available. Coordinate conversion is from one-based inclusive boxes to zero-based half-open PIL boxes. No additional crop context is used.
- Score each crop independently using a 2048→128→1 MLP with sigmoid. Inputs contain image and text features, their product, and absolute difference. The scorer never receives the target phrase, target index, or region label.
- Compare rank, rank+locality, rank+locality+control, a supervised omission classifier, text-only, image-only, inference-time cross-image feature shuffling, inverse cosine, negative text length, region area, and constant controls.
- Train for 60 epochs using AdamW (learning rate 0.001, weight decay 0.01), margin 0.1, unit locality/control weights, seeds 0/1/2. Select checkpoints every five epochs using validation intervention-target accuracy; retain the earliest maximum. This is a fixed initial configuration, not a hyperparameter search.
- Primary pilot ranking uses **score change** (q-minus minus q-plus), not absolute q-minus. Ties receive expected random tie-breaking accuracy and MRR. Metrics are averaged within image before aggregation. 95% intervals use 2,000 image bootstrap resamples. Paired differences use the same test images.
- Compare differently scaled methods primarily through rank accuracy and direction, not raw TIG. `region_area` is only a text-insensitive sanity control and its score scale is not calibrated.

## Artifacts

- `data/manifest.json`: exact captions, edits, boxes, image IDs, and split assignments.
- `data/provenance.json`: sources, selected image SHA256 values, selection seed, exclusions.
- `results/features.npz`: normalized features, indices, and manifest fingerprint.
- `results/feature_runtime.json`: actual extraction environment and time.
- `results/metrics.json`: all metrics and checkpoint choices, without result filtering.
- `results/per_image_metrics.json`: image-level paired analysis inputs.
- `results/predictions.npz`: raw per-region scores for every method.
- `results/checkpoints/`: fitted scorer states.
- `results/*.log`: preparation and execution logs.
- `results/final_metrics.json`: compact aggregate evidence across interventions, encoders, attribution methods, cluster-count ablations, and faithfulness checks.
- `../research/FINAL_REPORT.md`: generated claim-to-evidence report and measured compute ledger.

The feature cache fingerprint covers the manifest; changing the model version or preprocessing requires deleting the feature cache. Record model revision before a formal reproduction.

## Required before interpreting success

Independent human audit of edit validity; natural-caption and paraphrase tests; stronger region-caption/entailment or phrase matching baselines; an untouched semantic-fact test set; and confirmation that gains survive image shuffling controls. Attribute and relation claims require corresponding annotations. Pilot test results used to choose future directions make this an exploratory set; a subsequent confirmatory study needs a fresh held-out test set.

A further task-directed baseline, `max_ngram_cosine`, uses the maximum crop similarity over all 1–4 word contiguous spans of the current caption (excluding spans made entirely of a fixed small stopword set). Each span uses the prompt `a photo of {span}`. This baseline uses no target phrase, target ID, or annotation-based parsing at inference. It is a simple coverage baseline; it does not replace a full captioning-plus-entailment comparison.

## HPC status

The bounded Flickr30K experiment has run end to end on NYU HPC. The runner accepts `--device cuda`, `--batch-size`, `--data-dir`, `--output-dir`, `--minus-field`, and `--model`; both frozen feature extraction and small-scorer fitting use the selected CUDA device. See `../research/FINAL_REPORT.md` for results and limitations and `../hpc/RESOURCE_PLAN.md` for the resource policy.
