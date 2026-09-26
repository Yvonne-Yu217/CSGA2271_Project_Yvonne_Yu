# Final evidence report: What Does the Image Add?

## Summary

The core learned-method claim is **not supported** at this scale: the three-seed rank+locality+control seed mean reached Accuracy@1 `0.7722`, versus `0.8450` for inverse crop-text cosine. The shuffled-image control fell to `0.4326`, showing that the learned scorer uses visual input, but that does not establish an advantage over the simple baseline. RQ2 is **frozen transfer completed, but task validity is weak and superiority remains unsupported**. RQ3 remains partial: a SigLIP encoder check is included; automatic patch-cluster sensitivity is included, but Visual Genome and an independent region-proposal transfer benchmark are not.

This is a bounded validation, not a publication-ready confirmation. Deterministic and matched edits are synthetic; natural captions differ in multiple facts and remain noisy even after exact entity-ID and phrase exclusion.

## Validity limitations

The historical intervention and SigLIP scorers were independently retrained. The new frozen-checkpoint follow-up below is the actual transfer test. Natural captions may alter non-target facts; raw TIG/drift are scale-sensitive. Historical masked same-CLIP similarity is not independent factual recovery. See research/RESULTS_REVIEW_AND_NEXT_STEPS.md.

## Data and audit

- Flickr30K Entities: 350 images and 1165 intervention pairs; train/val/test images = 200/50/100.
- Natural-caption eligible pairs: 882; test = 251.
- Audit status: `pass`; verified image hashes = 350; errors = 0; warnings = 0.

- Semantic diagnostic review: 100 stratified rows; valid/invalid/uncertain = 44/53/3. All were reviewed by Codex from rendered images and captions, not by a human annotator; human-confirmed rows = 0.
- Diagnostic valid rows by intervention: tminus 10/33, tmatched 25/34, tnatural 9/33.
- Historical center-crop visibility: 190 full, 141 partial, 11 invisible regions; 19 regions used the nearest-patch fallback.

## Intervention results

All cells are image-level means with 95% bootstrap confidence intervals. Seeded rows average per-image metrics across three fixed seeds before the image bootstrap; this is not a prediction ensemble; seed-level Accuracy@1 values remain in `pilot/results/final_metrics.json`.

### Deterministic deletion

| Method | Acc@1 | MRR | TIG | Off-target drift |
|---|---:|---:|---:|---:|
| inverse_cosine | 0.8450 [0.8000, 0.8875] | 0.9141 [0.8882, 0.9379] | 0.0122 [0.0105, 0.0139] | 0.0057 [0.0051, 0.0063] |
| max_ngram_cosine | 0.7561 [0.7117, 0.7999] | 0.8661 [0.8404, 0.8913] | 0.0092 [0.0079, 0.0106] | 0.0005 [0.0003, 0.0006] |
| rank | 0.6342 [0.5867, 0.6842] | 0.8016 [0.7740, 0.8303] | 0.1336 [0.1208, 0.1466] | 0.1250 [0.1133, 0.1375] |
| rank+locality | 0.7708 [0.7208, 0.8158] | 0.8774 [0.8499, 0.9022] | 0.0110 [0.0098, 0.0123] | 0.0078 [0.0069, 0.0087] |
| rank+locality+control | 0.7722 [0.7264, 0.8153] | 0.8764 [0.8499, 0.9010] | 0.0041 [0.0036, 0.0046] | 0.0020 [0.0018, 0.0022] |
| shuffled-image control | 0.4326 [0.3819, 0.4847] | 0.6910 [0.6610, 0.7222] | 0.0014 [0.0011, 0.0017] | 0.0021 [0.0019, 0.0023] |
| supervised omission | 0.1725 [0.1291, 0.2175] | 0.5442 [0.5181, 0.5707] | 0.0179 [0.0155, 0.0204] | 0.0211 [0.0186, 0.0238] |
| text-only control | 0.4386 [0.4221, 0.4535] | 0.6954 [0.6802, 0.7094] | 0.0003 [0.0002, 0.0003] | 0.0005 [0.0004, 0.0005] |
| image-only control | 0.4386 [0.4221, 0.4535] | 0.6954 [0.6802, 0.7094] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| constant | 0.4386 [0.4221, 0.4536] | 0.6954 [0.6802, 0.7095] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

### Length-matched generalization

| Method | Acc@1 | MRR | TIG | Off-target drift |
|---|---:|---:|---:|---:|
| inverse_cosine | 0.7575 [0.7033, 0.8075] | 0.8666 [0.8363, 0.8942] | 0.0094 [0.0077, 0.0112] | 0.0059 [0.0053, 0.0065] |
| max_ngram_cosine | 0.7442 [0.7044, 0.7827] | 0.8587 [0.8348, 0.8806] | 0.0081 [0.0068, 0.0094] | 0.0004 [0.0003, 0.0006] |
| rank | 0.4558 [0.4042, 0.5067] | 0.7083 [0.6803, 0.7362] | 0.1588 [0.1461, 0.1723] | 0.1588 [0.1457, 0.1723] |
| rank+locality | 0.7406 [0.6936, 0.7864] | 0.8619 [0.8370, 0.8864] | 0.0279 [0.0254, 0.0303] | 0.0185 [0.0169, 0.0201] |
| rank+locality+control | 0.7486 [0.7014, 0.7917] | 0.8653 [0.8400, 0.8886] | 0.0102 [0.0092, 0.0113] | 0.0066 [0.0059, 0.0073] |
| shuffled-image control | 0.4824 [0.4318, 0.5346] | 0.7171 [0.6883, 0.7468] | 0.0061 [0.0054, 0.0068] | 0.0061 [0.0055, 0.0067] |
| supervised omission | 0.2156 [0.1761, 0.2567] | 0.5649 [0.5397, 0.5895] | 0.0301 [0.0267, 0.0337] | 0.0331 [0.0295, 0.0368] |
| text-only control | 0.4386 [0.4221, 0.4535] | 0.6954 [0.6802, 0.7094] | 0.0016 [0.0014, 0.0017] | 0.0016 [0.0015, 0.0018] |
| image-only control | 0.4386 [0.4221, 0.4535] | 0.6954 [0.6802, 0.7094] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| constant | 0.4386 [0.4221, 0.4536] | 0.6954 [0.6802, 0.7095] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

### Natural same-image caption

| Method | Acc@1 | MRR | TIG | Off-target drift |
|---|---:|---:|---:|---:|
| inverse_cosine | 0.5816 [0.5198, 0.6435] | 0.7648 [0.7264, 0.8018] | 0.0133 [0.0104, 0.0160] | 0.0145 [0.0130, 0.0160] |
| max_ngram_cosine | 0.5902 [0.5284, 0.6521] | 0.7719 [0.7362, 0.8070] | 0.0092 [0.0071, 0.0112] | 0.0097 [0.0083, 0.0110] |
| rank | 0.4404 [0.3837, 0.4931] | 0.6966 [0.6642, 0.7273] | 0.0741 [0.0571, 0.0920] | 0.1047 [0.0927, 0.1177] |
| rank+locality | 0.5066 [0.4542, 0.5601] | 0.7306 [0.7001, 0.7620] | 0.0040 [0.0030, 0.0051] | 0.0061 [0.0054, 0.0067] |
| rank+locality+control | 0.5017 [0.4490, 0.5550] | 0.7263 [0.6960, 0.7568] | 0.0028 [0.0020, 0.0037] | 0.0048 [0.0043, 0.0054] |
| shuffled-image control | 0.3810 [0.3144, 0.4410] | 0.6580 [0.6193, 0.6936] | 0.0010 [0.0001, 0.0019] | 0.0047 [0.0042, 0.0053] |
| supervised omission | 0.3130 [0.2583, 0.3691] | 0.6235 [0.5906, 0.6551] | 0.0263 [0.0197, 0.0334] | 0.0414 [0.0369, 0.0461] |
| text-only control | 0.4334 [0.4153, 0.4495] | 0.6907 [0.6739, 0.7056] | 0.0001 [-0.0000, 0.0003] | 0.0009 [0.0008, 0.0010] |
| image-only control | 0.4334 [0.4153, 0.4495] | 0.6907 [0.6739, 0.7056] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| constant | 0.4334 [0.4160, 0.4505] | 0.6907 [0.6740, 0.7066] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Frozen deletion-checkpoint transfer

The same three deletion-trained checkpoints are evaluated without optimization on every intervention. The prediction ensemble averages scores before ranking; `seed metric mean` averages the three independently computed metrics. The reviewed-valid subset is a small Codex visual diagnostic, not human gold data.

| Intervention | Subset | Pairs | Method | Delta Acc@1 | Absolute Acc@1 |
|---|---|---:|---|---:|---:|
| tminus | all_exploratory | 337 | inverse_cosine | 0.8450 [0.7992, 0.8892] | 0.6050 [0.5575, 0.6517] |
| tminus | all_exploratory | 337 | seed_metric_mean | 0.7722 [0.7264, 0.8153] | 0.6133 [0.5700, 0.6550] |
| tminus | all_exploratory | 337 | prediction_ensemble | 0.7908 [0.7400, 0.8358] | 0.6233 [0.5725, 0.6700] |
| tminus | audited_valid | 10 | inverse_cosine | 0.8333 [0.6111, 1.0000] | 0.5000 [0.2222, 0.8333] |
| tminus | audited_valid | 10 | seed_metric_mean | 0.8704 [0.6852, 1.0000] | 0.5741 [0.3519, 0.7963] |
| tminus | audited_valid | 10 | prediction_ensemble | 0.8889 [0.6667, 1.0000] | 0.6111 [0.3319, 0.8889] |
| tmatched | all_exploratory | 337 | inverse_cosine | 0.7575 [0.7000, 0.8059] | 0.6167 [0.5658, 0.6633] |
| tmatched | all_exploratory | 337 | seed_metric_mean | 0.7183 [0.6675, 0.7661] | 0.5811 [0.5386, 0.6231] |
| tmatched | all_exploratory | 337 | prediction_ensemble | 0.7317 [0.6792, 0.7809] | 0.5988 [0.5483, 0.6454] |
| tmatched | audited_valid | 25 | inverse_cosine | 0.6364 [0.4091, 0.8182] | 0.6818 [0.5000, 0.8636] |
| tmatched | audited_valid | 25 | seed_metric_mean | 0.6742 [0.5303, 0.8182] | 0.4545 [0.2803, 0.6288] |
| tmatched | audited_valid | 25 | prediction_ensemble | 0.6818 [0.5000, 0.8636] | 0.5000 [0.3170, 0.7273] |
| tnatural | all_exploratory | 251 | inverse_cosine | 0.5816 [0.5180, 0.6409] | 0.5550 [0.4871, 0.6186] |
| tnatural | all_exploratory | 251 | seed_metric_mean | 0.5515 [0.4903, 0.6114] | 0.5137 [0.4565, 0.5682] |
| tnatural | all_exploratory | 251 | prediction_ensemble | 0.5601 [0.4957, 0.6220] | 0.5266 [0.4579, 0.5885] |
| tnatural | audited_valid | 9 | inverse_cosine | 0.5556 [0.2222, 0.8889] | 0.7778 [0.4444, 1.0000] |
| tnatural | audited_valid | 9 | seed_metric_mean | 0.6296 [0.3333, 0.8889] | 0.4815 [0.1481, 0.7778] |
| tnatural | audited_valid | 9 | prediction_ensemble | 0.5556 [0.2222, 0.8889] | 0.4444 [0.1111, 0.7778] |

### Frozen ensemble paired against inverse cosine

| Intervention | Subset | Delta Acc@1 difference | Absolute Acc@1 difference |
|---|---|---:|---:|
| tminus | all_exploratory | -0.0542 [-0.1042, -0.0067] | 0.0183 [-0.0350, 0.0725] |
| tminus | audited_valid | 0.0556 [-0.1667, 0.3333] | 0.1111 [-0.2222, 0.4444] |
| tmatched | all_exploratory | -0.0258 [-0.0792, 0.0275] | -0.0179 [-0.0750, 0.0413] |
| tmatched | audited_valid | 0.0455 [-0.0909, 0.1818] | -0.1818 [-0.3864, 0.0227] |
| tnatural | all_exploratory | -0.0215 [-0.0498, 0.0060] | -0.0284 [-0.1023, 0.0412] |
| tnatural | audited_valid | 0.0000 [-0.3333, 0.3333] | -0.3333 [-0.6667, 0.0000] |

## Spatial attribution proxies

These are held-out, task-adapted Hugging Face CLIP implementations. Grad-ECLIP uses the published final-attention gradient/value/key-similarity formula; CCI uses cosine patch clustering and final-block CLS-to-cluster attention masking. They are algorithm adaptations, not execution of the authors’ repositories.

| Method | Acc@1 | MRR | TIG | Off-target drift |
|---|---:|---:|---:|---:|
| pooled_patch_inverse_cosine | 0.1292 [0.0925, 0.1692] | 0.5173 [0.4938, 0.5417] | -0.0120 [-0.0134, -0.0107] | 0.0081 [0.0072, 0.0091] |
| region_masking_inverse_importance | 0.7492 [0.6975, 0.8000] | 0.8628 [0.8337, 0.8918] | 0.0163 [0.0135, 0.0193] | 0.0061 [0.0052, 0.0071] |
| cci_cluster_inverse_importance | 0.7204 [0.6667, 0.7717] | 0.8441 [0.8135, 0.8737] | 0.0073 [0.0060, 0.0088] | 0.0036 [0.0031, 0.0042] |
| grad_eclip_inverse_saliency | 0.7654 [0.7133, 0.8146] | 0.8732 [0.8439, 0.9003] | 0.0158 [0.0130, 0.0186] | 0.0061 [0.0052, 0.0070] |

### Spatial attribution intervention robustness

| Intervention | Method | Acc@1 | MRR |
|---|---|---:|---:|
| tminus | region_masking_inverse_importance | 0.7492 [0.6975, 0.8000] | 0.8628 [0.8337, 0.8918] |
| tminus | cci_cluster_inverse_importance | 0.7204 [0.6667, 0.7717] | 0.8441 [0.8135, 0.8737] |
| tminus | grad_eclip_inverse_saliency | 0.7654 [0.7133, 0.8146] | 0.8732 [0.8439, 0.9003] |
| tmatched | region_masking_inverse_importance | 0.6967 [0.6433, 0.7483] | 0.8334 [0.8050, 0.8619] |
| tmatched | cci_cluster_inverse_importance | 0.7033 [0.6483, 0.7550] | 0.8343 [0.8022, 0.8636] |
| tmatched | grad_eclip_inverse_saliency | 0.7612 [0.7083, 0.8100] | 0.8704 [0.8397, 0.8976] |
| tnatural | region_masking_inverse_importance | 0.5619 [0.4974, 0.6255] | 0.7551 [0.7177, 0.7919] |
| tnatural | cci_cluster_inverse_importance | 0.5490 [0.4905, 0.6057] | 0.7517 [0.7182, 0.7839] |
| tnatural | grad_eclip_inverse_saliency | 0.5945 [0.5361, 0.6538] | 0.7748 [0.7402, 0.8091] |

### Automatic patch-cluster count ablation

| CCI clusters | Acc@1 | MRR |
|---:|---:|---:|
| 3 | 0.6958 [0.6421, 0.7467] | 0.8323 [0.8032, 0.8614] |
| 8 | 0.7558 [0.7016, 0.8067] | 0.8645 [0.8335, 0.8935] |
| 12 | 0.7675 [0.7141, 0.8150] | 0.8729 [0.8421, 0.8996] |

## Encoder replication: independently retrained SigLIP

| Intervention | Method | Acc@1 | MRR | TIG | Off-target drift |
|---|---|---:|---:|---:|---:|
| tminus | inverse_cosine | 0.6117 [0.5533, 0.6692] | 0.7859 [0.7512, 0.8198] | 0.0051 [0.0023, 0.0081] | 0.0151 [0.0131, 0.0174] |
| tminus | max_ngram_cosine | 0.5031 [0.4577, 0.5520] | 0.7276 [0.6994, 0.7571] | 0.0016 [0.0012, 0.0020] | 0.0014 [0.0011, 0.0017] |
| tminus | rank+locality+control | 0.5142 [0.4747, 0.5519] | 0.7365 [0.7127, 0.7591] | 0.0005 [0.0004, 0.0006] | 0.0007 [0.0006, 0.0007] |
| tminus | shuffled-image control | 0.4690 [0.4197, 0.5176] | 0.7108 [0.6817, 0.7389] | 0.0004 [0.0003, 0.0005] | 0.0007 [0.0006, 0.0007] |
| tmatched | inverse_cosine | 0.5758 [0.5225, 0.6317] | 0.7677 [0.7363, 0.7991] | 0.0068 [0.0039, 0.0097] | 0.0136 [0.0118, 0.0154] |
| tmatched | max_ngram_cosine | 0.4515 [0.4095, 0.4947] | 0.7032 [0.6785, 0.7271] | 0.0004 [-0.0002, 0.0010] | 0.0018 [0.0015, 0.0022] |
| tmatched | rank+locality+control | 0.5528 [0.5092, 0.5964] | 0.7558 [0.7299, 0.7813] | 0.0042 [0.0036, 0.0048] | 0.0038 [0.0033, 0.0043] |
| tmatched | shuffled-image control | 0.4568 [0.4001, 0.5081] | 0.7052 [0.6706, 0.7349] | 0.0030 [0.0024, 0.0036] | 0.0037 [0.0032, 0.0041] |
| tnatural | inverse_cosine | 0.4759 [0.4124, 0.5404] | 0.7102 [0.6711, 0.7498] | 0.0007 [-0.0043, 0.0056] | 0.0234 [0.0202, 0.0271] |
| tnatural | max_ngram_cosine | 0.4573 [0.3940, 0.5206] | 0.7021 [0.6679, 0.7372] | 0.0028 [0.0016, 0.0040] | 0.0067 [0.0059, 0.0076] |
| tnatural | rank+locality+control | 0.4522 [0.4021, 0.5006] | 0.7048 [0.6762, 0.7314] | 0.0006 [0.0003, 0.0009] | 0.0014 [0.0012, 0.0016] |
| tnatural | shuffled-image control | 0.4366 [0.3809, 0.4920] | 0.6942 [0.6621, 0.7257] | 0.0005 [0.0003, 0.0008] | 0.0013 [0.0012, 0.0015] |

## Absolute localization and omitted-phrase recoverability

Recoverability is the CLIP similarity drop for the target phrase after mean-color masking the method-selected region.

| Intervention | Method | Absolute localization Acc@1 | Localization MRR | Phrase-similarity drop |
|---|---|---:|---:|---:|
| tminus | learned_rank_local_control | 0.6233 [0.5742, 0.6733] | 0.7956 [0.7672, 0.8238] | 0.0012 [-0.0020, 0.0045] |
| tminus | inverse_cosine | 0.6050 [0.5550, 0.6517] | 0.7879 [0.7593, 0.8145] | 0.0032 [0.0002, 0.0064] |
| tminus | max_ngram_cosine | 0.6717 [0.6250, 0.7200] | 0.8255 [0.7999, 0.8511] | 0.0027 [-0.0005, 0.0062] |
| tminus | region_area | 0.4667 [0.4317, 0.5033] | 0.7064 [0.6853, 0.7270] | -0.0033 [-0.0061, -0.0004] |
| tminus | random | 0.4700 [0.4142, 0.5283] | 0.7105 [0.6775, 0.7435] | -0.0023 [-0.0055, 0.0008] |
| tminus | ground_truth_target | - | - | 0.0067 [0.0034, 0.0101] |
| tmatched | learned_rank_local_control | 0.6325 [0.5850, 0.6825] | 0.7987 [0.7712, 0.8258] | 0.0013 [-0.0019, 0.0047] |
| tmatched | inverse_cosine | 0.6167 [0.5650, 0.6675] | 0.7945 [0.7652, 0.8224] | 0.0029 [-0.0001, 0.0059] |
| tmatched | max_ngram_cosine | 0.6617 [0.6150, 0.7083] | 0.8217 [0.7970, 0.8466] | 0.0021 [-0.0010, 0.0055] |
| tmatched | region_area | 0.4667 [0.4317, 0.5033] | 0.7064 [0.6853, 0.7270] | -0.0033 [-0.0061, -0.0004] |
| tmatched | random | 0.4700 [0.4142, 0.5283] | 0.7105 [0.6775, 0.7435] | -0.0023 [-0.0055, 0.0008] |
| tmatched | ground_truth_target | - | - | 0.0067 [0.0034, 0.0101] |
| tnatural | learned_rank_local_control | 0.4820 [0.4158, 0.5481] | 0.7154 [0.6793, 0.7512] | -0.0035 [-0.0070, 0.0003] |
| tnatural | inverse_cosine | 0.5550 [0.4923, 0.6212] | 0.7559 [0.7190, 0.7931] | -0.0002 [-0.0033, 0.0033] |
| tnatural | max_ngram_cosine | 0.6151 [0.5515, 0.6787] | 0.7867 [0.7522, 0.8219] | -0.0004 [-0.0037, 0.0029] |
| tnatural | region_area | 0.4244 [0.3625, 0.4854] | 0.6792 [0.6451, 0.7128] | -0.0076 [-0.0111, -0.0039] |
| tnatural | random | 0.4321 [0.3634, 0.5026] | 0.6880 [0.6500, 0.7272] | -0.0045 [-0.0077, -0.0010] |
| tnatural | ground_truth_target | - | - | 0.0033 [-0.0003, 0.0071] |

## Claim-to-evidence audit

| Claim | Evidence | Status |
|---|---|---|
| RQ1: learned region complementarity is measurable | Positive controlled gaps and visual-shuffle degradation, but learned Acc@1 `0.7722` does not beat inverse cosine `0.8450` | Mixed / primary superiority claim not supported |
| RQ2: response is local under text intervention | Frozen transfer is complete; only 44/100 diagnostic rows passed all semantic checks, with no human confirmation | Task validity weak; no confirmatory claim |
| RQ3: transfer across datasets/encoders/regions | SigLIP independently retrained encoder check completed; automatic patch-cluster sensitivity completed; Visual Genome and independent region proposals not completed | Partially tested |
| Existing importance methods solve complementarity | Algorithm-adapted CCI and Grad-ECLIP metrics are reported on the same boxes | Adaptation evidence; authors’ repository execution not performed |

## Failure taxonomy

- Strong simple baseline: inverse crop-text cosine is stronger than the learned scorer on controlled deletion.
- Natural-caption difficulty: independently retrained performance is lower for natural captions, which change more than the target entity.
- Locality/strength tradeoff: rank-only training creates large TIG but also large off-target drift; locality losses reduce both.
- Attribution mismatch: pooled-patch, CCI, and Grad-ECLIP importance need not encode “unmentioned visual content.”
- Annotation validity: exact phrase/ID filtering cannot rule out synonymous or implicit target mention in natural captions.

## Decision gate

The gate selects **finish the rigorous course comparison and repair or replace the task before further learned-method development**. The frozen ensemble does not beat inverse cosine on the full deletion benchmark, while fewer than half of the stratified semantic-review rows pass all checks. No learned variant, untouched-set confirmation, Visual Genome expansion, or CVPR experiment is authorized by this evidence. Human review remains desirable, but cannot rescue claims from the current labels without a revised task.

## Compute ledger

| Job | Partition | Elapsed (s) | GPU-hours | State |
|---|---|---:|---:|---|
| 1931 | c12m85-a100-1 | 104 | 0.0289 | COMPLETED |
| 1932 | c12m85-a100-1 | 69 | 0.0192 | COMPLETED |
| 1933 | c12m85-a100-1 | 68 | 0.0189 | COMPLETED |
| 1934 | c12m85-a100-1 | 68 | 0.0189 | COMPLETED |
| 1935 | c12m85-a100-1 | 0 | 0.0000 | CANCELLED by 4645820 |
| 1957 | c12m85-a100-1 | 1326 | 0.3683 | CANCELLED by 0 |
| 1956 | g2-standard-12 | 373 | 0.1036 | CANCELLED by 4645820 |
| 1960 | g2-standard-12 | 189 | 0.0525 | COMPLETED |
| 1961 | g2-standard-12 | 41 | 0.0114 | FAILED |
| 1964 | g2-standard-12 | 0 | 0.0000 | CANCELLED by 4645820 |
| 1966 | g2-standard-12 | 70 | 0.0194 | COMPLETED |
| 1967 | g2-standard-12 | 39 | 0.0108 | COMPLETED |
| 1969 | g2-standard-12 | 1719 | 0.4775 | CANCELLED by 4645820 |
| 1971 | g2-standard-12 | 41 | 0.0114 | COMPLETED |
| 1972 | g2-standard-12 | 683 | 0.1897 | CANCELLED by 4645820 |
| 1973 | g2-standard-12 | 304 | 0.0844 | CANCELLED by 4645820 |
| 1974 | g2-standard-12 | 1478 | 0.4106 | CANCELLED by 4645820 |
| 1975 | g2-standard-12 | 748 | 0.2078 | RUNNING |

## Utilization monitoring

| Run | Samples | Mean GPU util. | Max GPU util. | Nonzero samples |
|---|---:|---:|---:|---:|
| formal_l4 | 19 | 16.2% | 100% | 36.8% |
| siglip_l4 | 7 | 15.9% | 59% | 28.6% |
| faithfulness_l4 | 20 | 3.9% | 73% | 10.0% |
| frozen_followup | 8 | 11.6% | 91% | 25.0% |

Recorded allocation total at report generation: `2.0333` GPU-hours. This snapshot is not a final authorization balance; reconcile final cumulative usage under GOAL.md before submission. The currently running notebook allocation is server-managed under the latest user directive and will not be manually canceled; its eventual final state and elapsed time remain recoverable from sacct.

## Reproduction

```sh
python3 -u pilot/fetch_annotations.py
python3 -u pilot/prepare.py --counts 200,50,100
python3 -u pilot/audit.py
sbatch hpc/full_l4.sbatch
python3 -u pilot/next_round.py preflight --source SOURCE_TMINUS --output FOLLOWUP_DIR
python3 -u pilot/visibility_audit.py --manifest SOURCE_TMINUS/manifest_snapshot.json --output FOLLOWUP_DIR/visibility_audit.json
python3 -u pilot/next_round.py gpu --source SOURCE_TMINUS --output FOLLOWUP_DIR
python3 -u pilot/render_semantic_audit.py --audit FOLLOWUP_DIR/semantic_audit.csv --output FOLLOWUP_DIR/audit-pages
python3 -u pilot/next_round.py summarize --output FOLLOWUP_DIR
python3 -u pilot/make_report.py --formal-root pilot/results/formal-l4-JOB_ID --job-ids JOB_ID
```

The dataset, model cache, raw predictions, and checkpoints are intentionally excluded from Git. Compact audited metrics and this report are committed.
