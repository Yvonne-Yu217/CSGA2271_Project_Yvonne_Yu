"""Create the compact evidence bundle and final Markdown report."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from run import NAMES, summarize


FAMILIES = {
    'rank': ['rank_seed0', 'rank_seed1', 'rank_seed2'],
    'rank+locality': ['rank_local_seed0', 'rank_local_seed1', 'rank_local_seed2'],
    'rank+locality+control': ['rank_local_control_seed0', 'rank_local_control_seed1', 'rank_local_control_seed2'],
    'shuffled-image control': ['shuffled_image_seed0', 'shuffled_image_seed1', 'shuffled_image_seed2'],
    'supervised omission': ['supervised_omission_seed0', 'supervised_omission_seed1', 'supervised_omission_seed2'],
    'text-only control': ['text_only_seed0', 'text_only_seed1', 'text_only_seed2'],
    'image-only control': ['image_only_seed0', 'image_only_seed1', 'image_only_seed2'],
}
DIRECT = ['inverse_cosine', 'max_ngram_cosine', 'region_area', 'negative_text_length', 'constant']


def aggregate(metrics_path):
    metrics = json.loads(metrics_path.read_text())
    per_path = metrics_path.with_name('per_image_metrics.json')
    per = json.loads(per_path.read_text())
    result = {'metadata': {k: metrics[k] for k in ['scope', 'intervention', 'image_counts', 'pair_counts']}, 'methods': {}}
    inverse = {key: np.array(value) for key, value in per['inverse_cosine'].items()}
    for name in DIRECT:
        result['methods'][name] = metrics['metrics'][name]
    for family, members in FAMILIES.items():
        common = sorted(set.intersection(*(set(per[name]) for name in members)))
        seed_mean = {key: np.mean([np.array(per[name][key]) for name in members], axis=0) for key in common}
        family_summary = summarize(seed_mean)
        seed_acc = [metrics['metrics'][name]['delta_acc1']['mean'] for name in members]
        paired = {key: seed_mean[key] - inverse[key] for key in common}
        family_summary['seed_delta_acc1'] = seed_acc
        family_summary['seed_delta_acc1_sd'] = float(np.std(seed_acc, ddof=1))
        family_summary['paired_delta_acc1_vs_inverse'] = summarize(paired)['delta_acc1']
        result['methods'][family] = family_summary
    return result


def metric_cell(method, metric):
    item = method[metric]
    return f"{item['mean']:.4f} [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"


def followup_cell(method, metric):
    return metric_cell(method['metrics'], metric)


def ledger(job_ids):
    if not job_ids:
        return []
    command = ['sacct', '-j', ','.join(job_ids), '-X', '-n', '-P', '--format=JobIDRaw,Partition,ElapsedRaw,State,AllocTRES']
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = []
    if proc.returncode:
        return [{'error': proc.stderr.strip()}]
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        job, partition, seconds, state, tres = line.split('|')[:5]
        if '.' in job:
            continue
        allocations = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
        if 'gres/gpu' in allocations:
            gpu_count = int(allocations['gres/gpu'])
        else:
            gpu_count = sum(int(value) for key, value in allocations.items() if key.startswith('gres/gpu:'))
        rows.append({'job_id': job, 'partition': partition, 'elapsed_seconds': int(seconds or 0),
                     'allocated_gpus': gpu_count, 'gpu_hours': gpu_count * int(seconds or 0) / 3600,
                     'state': state, 'alloc_tres': tres})
    return rows


def utilization_summary(path):
    samples = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        match = re.search(r' (\d+) %,', line)
        if match:
            samples.append(int(match.group(1)))
    return {'samples': len(samples), 'mean_gpu_utilization_percent': float(np.mean(samples)) if samples else None,
            'max_gpu_utilization_percent': max(samples) if samples else None,
            'nonzero_sample_fraction': float(np.mean(np.array(samples) > 0)) if samples else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal-root', type=Path, required=True)
    ap.add_argument('--audit', type=Path, default=Path(__file__).resolve().parent / 'results' / 'data_audit.json')
    ap.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1] / 'research' / 'FINAL_REPORT.md')
    ap.add_argument('--compact', type=Path, default=Path(__file__).resolve().parent / 'results' / 'final_metrics.json')
    ap.add_argument('--job-ids', default='')
    ap.add_argument('--siglip-dir', type=Path)
    ap.add_argument('--siglip-matched-dir', type=Path)
    ap.add_argument('--siglip-natural-dir', type=Path)
    ap.add_argument('--faithfulness-dir', type=Path)
    ap.add_argument('--faithfulness-matched-dir', type=Path)
    ap.add_argument('--faithfulness-natural-dir', type=Path)
    ap.add_argument('--spatial-dir', type=Path)
    ap.add_argument('--spatial-matched-dir', type=Path)
    ap.add_argument('--spatial-natural-dir', type=Path)
    ap.add_argument('--spatial-cluster-dir', type=Path, action='append', default=[])
    ap.add_argument('--frozen-followup-dir', type=Path)
    args = ap.parse_args()
    audit = json.loads(args.audit.read_text())
    interventions = {field: aggregate(args.formal_root / field / 'metrics.json') for field in ['tminus', 'tmatched', 'tnatural']}
    spatial_dir = args.spatial_dir or (args.formal_root / 'spatial-baselines')
    spatial = {'tminus': json.loads((spatial_dir / 'metrics.json').read_text())}
    for field, directory in [('tmatched', args.spatial_matched_dir), ('tnatural', args.spatial_natural_dir)]:
        if directory:
            spatial[field] = json.loads((directory / 'metrics.json').read_text())
    cluster_ablation = [json.loads((directory / 'metrics.json').read_text())
                        for directory in args.spatial_cluster_dir]
    siglip = {}
    for field, directory in [('tminus', args.siglip_dir), ('tmatched', args.siglip_matched_dir),
                             ('tnatural', args.siglip_natural_dir)]:
        if directory:
            siglip[field] = aggregate(directory / 'metrics.json')
    faithfulness = {}
    for field, directory in [('tminus', args.faithfulness_dir),
                             ('tmatched', args.faithfulness_matched_dir),
                             ('tnatural', args.faithfulness_natural_dir)]:
        if directory:
            faithfulness[field] = json.loads((directory / 'metrics.json').read_text())
    frozen = json.loads((args.frozen_followup_dir / 'summary.json').read_text()) if args.frozen_followup_dir else None
    visibility = json.loads((args.frozen_followup_dir / 'visibility_audit.json').read_text()) if args.frozen_followup_dir else None
    compute = ledger([value for value in args.job_ids.split(',') if value])
    utilization = {'formal_l4': utilization_summary(args.formal_root / 'utilization.csv')}
    if args.siglip_dir:
        utilization['siglip_l4'] = utilization_summary(args.siglip_dir / 'utilization.csv')
    if args.faithfulness_dir:
        utilization['faithfulness_l4'] = utilization_summary(args.faithfulness_dir / 'utilization.csv')
    if args.frozen_followup_dir:
        traces = sorted(args.frozen_followup_dir.glob('utilization-*.log'))
        if traces:
            utilization['frozen_followup'] = utilization_summary(traces[-1])
    compact = {'data_audit': audit, 'interventions': interventions, 'spatial_baselines': spatial,
               'spatial_cluster_ablation': cluster_ablation,
               'faithfulness': faithfulness, 'encoder_transfer': {'siglip': siglip} if siglip else {},
               'frozen_checkpoint_followup': frozen or {},
               'visibility_audit': ({k: visibility[k] for k in ('scope', 'images', 'unique_regions',
                   'region_visibility_counts', 'nearest_patch_fallback_regions', 'target_pair_visibility_counts')}
                   if visibility else {}),
               'utilization': utilization, 'compute_ledger': compute,
               'interpretation': {'encoder_transfer_key_is_legacy': True,
                   'siglip_protocol': 'independently retrained, not frozen transfer',
                   'seeded_intervention_rows': 'mean of per-image seed metrics, not prediction ensemble'}}
    args.compact.parent.mkdir(parents=True, exist_ok=True)
    args.compact.write_text(json.dumps(compact, indent=2))

    det = interventions['tminus']['methods']
    learned = det['rank+locality+control']['delta_acc1']['mean']
    inverse = det['inverse_cosine']['delta_acc1']['mean']
    shuffle = det['shuffled-image control']['delta_acc1']['mean']
    natural = interventions['tnatural']['methods']['rank+locality+control']['delta_acc1']['mean']
    rq1 = 'not supported' if learned <= inverse else 'supported on the controlled benchmark'
    rq2 = ('frozen transfer completed, but task validity is weak and superiority remains unsupported'
           if frozen else ('controlled-edit evidence only; semantic locality and frozen transfer remain unresolved'
                           if learned > shuffle and natural < learned else 'mixed'))
    lines = [
        '# Final evidence report: What Does the Image Add?', '',
        '## Summary', '',
        f'The core learned-method claim is **{rq1}** at this scale: the three-seed rank+locality+control seed mean reached '
        f'Accuracy@1 `{learned:.4f}`, versus `{inverse:.4f}` for inverse crop-text cosine. The shuffled-image control fell to '
        f'`{shuffle:.4f}`, showing that the learned scorer uses visual input, but that does not establish an advantage over the simple baseline. '
        f"RQ2 is **{rq2}**. RQ3 remains partial: {'a SigLIP encoder check is included; ' if siglip else ''}"
        f"automatic patch-cluster sensitivity is included, but Visual Genome and an independent region-proposal transfer benchmark are not.", '',
        'This is a bounded validation, not a publication-ready confirmation. Deterministic and matched edits are synthetic; natural captions differ in multiple facts and remain noisy even after exact entity-ID and phrase exclusion.', '',
        '## Validity limitations', '',
        'The historical intervention and SigLIP scorers were independently retrained. The new frozen-checkpoint follow-up below is the actual transfer test. Natural captions may alter non-target facts; raw TIG/drift are scale-sensitive. Historical masked same-CLIP similarity is not independent factual recovery. See research/RESULTS_REVIEW_AND_NEXT_STEPS.md.', '',
        '## Data and audit', '',
        f"- Flickr30K Entities: {audit['total_images']} images and {audit['total_pairs']} intervention pairs; "
        f"train/val/test images = {audit['image_counts']['train']}/{audit['image_counts']['val']}/{audit['image_counts']['test']}.",
        f"- Natural-caption eligible pairs: {sum(audit['natural_pair_counts'].values())}; test = {audit['natural_pair_counts']['test']}.",
        f"- Audit status: `{audit['status']}`; verified image hashes = {audit['verified_image_hashes']}; errors = {len(audit['errors'])}; warnings = {len(audit['warnings'])}.", '',
        '## Intervention results', '',
        'All cells are image-level means with 95% bootstrap confidence intervals. Seeded rows average per-image metrics across three fixed seeds before the image bootstrap; this is not a prediction ensemble; seed-level Accuracy@1 values remain in `pilot/results/final_metrics.json`.', ''
    ]
    audit_insert = lines.index('## Intervention results')
    audit_details = []
    if frozen:
        audit_details += [f"- Semantic diagnostic review: {sum(frozen['audit_counts'].values())} stratified rows; "
                          f"valid/invalid/uncertain = {frozen['audit_counts'].get('valid', 0)}/"
                          f"{frozen['audit_counts'].get('invalid', 0)}/{frozen['audit_counts'].get('uncertain', 0)}. "
                          f"All were reviewed by Codex from rendered images and captions, not by a human annotator; "
                          f"human-confirmed rows = {frozen['human_confirmed_valid_rows']}."]
        per_field = frozen['audit_counts_by_intervention']
        audit_details += ["- Diagnostic valid rows by intervention: " + ', '.join(
            f"{field} {values.get('valid', 0)}/{sum(values.values())}"
            for field, values in per_field.items()) + '.']
    if visibility:
        counts = visibility['region_visibility_counts']
        audit_details += [f"- Historical center-crop visibility: {counts['full']} full, {counts['partial']} partial, "
                          f"{counts['invisible']} invisible regions; {visibility['nearest_patch_fallback_regions']} "
                          f"regions used the nearest-patch fallback."]
    if audit_details:
        lines[audit_insert:audit_insert] = audit_details + ['']
    for field, title in [('tminus', 'Deterministic deletion'), ('tmatched', 'Length-matched generalization'), ('tnatural', 'Natural same-image caption')]:
        lines += [f'### {title}', '', '| Method | Acc@1 | MRR | TIG | Off-target drift |', '|---|---:|---:|---:|---:|']
        methods = interventions[field]['methods']
        order = ['inverse_cosine', 'max_ngram_cosine', 'rank', 'rank+locality', 'rank+locality+control',
                 'shuffled-image control', 'supervised omission', 'text-only control', 'image-only control', 'constant']
        for method in order:
            value = methods[method]
            lines.append(f"| {method} | {metric_cell(value, 'delta_acc1')} | {metric_cell(value, 'delta_mrr')} | {metric_cell(value, 'TIG')} | {metric_cell(value, 'off_target_drift')} |")
        lines.append('')
    if frozen:
        lines += ['## Frozen deletion-checkpoint transfer', '',
                  'The same three deletion-trained checkpoints are evaluated without optimization on every intervention. '
                  'The prediction ensemble averages scores before ranking; `seed metric mean` averages the three independently computed metrics. '
                  'The reviewed-valid subset is a small Codex visual diagnostic, not human gold data.', '',
                  '| Intervention | Subset | Pairs | Method | Delta Acc@1 | Absolute Acc@1 |',
                  '|---|---|---:|---|---:|---:|']
        for field in ['tminus', 'tmatched', 'tnatural']:
            for subset in ['all_exploratory', 'audited_valid']:
                result = frozen['interventions'][field][subset]
                for method in ['inverse_cosine', 'seed_metric_mean', 'prediction_ensemble']:
                    value = result['methods'][method]
                    lines.append(f"| {field} | {subset} | {result['pairs']} | {method} | "
                                 f"{followup_cell(value, 'delta_acc1')} | {followup_cell(value, 'absolute_acc1')} |")
        lines += ['', '### Frozen ensemble paired against inverse cosine', '',
                  '| Intervention | Subset | Delta Acc@1 difference | Absolute Acc@1 difference |',
                  '|---|---|---:|---:|']
        for field in ['tminus', 'tmatched', 'tnatural']:
            for subset in ['all_exploratory', 'audited_valid']:
                value = frozen['interventions'][field][subset]['paired_vs_inverse_cosine']['prediction_ensemble']
                lines.append(f"| {field} | {subset} | {followup_cell(value, 'delta_acc1')} | "
                             f"{followup_cell(value, 'absolute_acc1')} |")
        lines.append('')
    lines += ['## Spatial attribution proxies', '',
        'These are held-out, task-adapted Hugging Face CLIP implementations. Grad-ECLIP uses the published final-attention gradient/value/key-similarity formula; CCI uses cosine patch clustering and final-block CLS-to-cluster attention masking. They are algorithm adaptations, not execution of the authors’ repositories.', '',
              '| Method | Acc@1 | MRR | TIG | Off-target drift |', '|---|---:|---:|---:|---:|']
    for method, value in spatial['tminus']['metrics'].items():
        lines.append(f"| {method} | {metric_cell(value, 'delta_acc1')} | {metric_cell(value, 'delta_mrr')} | {metric_cell(value, 'TIG')} | {metric_cell(value, 'off_target_drift')} |")
    if len(spatial) > 1:
        lines += ['', '### Spatial attribution intervention robustness', '',
                  '| Intervention | Method | Acc@1 | MRR |', '|---|---|---:|---:|']
        for field, result in spatial.items():
            for method in ['region_masking_inverse_importance', 'cci_cluster_inverse_importance',
                           'grad_eclip_inverse_saliency']:
                value = result['metrics'][method]
                lines.append(f"| {field} | {method} | {metric_cell(value, 'delta_acc1')} | {metric_cell(value, 'delta_mrr')} |")
    if cluster_ablation:
        lines += ['', '### Automatic patch-cluster count ablation', '',
                  '| CCI clusters | Acc@1 | MRR |', '|---:|---:|---:|']
        for result in cluster_ablation:
            value = result['metrics']['cci_cluster_inverse_importance']
            lines.append(f"| {result['cci_clusters']} | {metric_cell(value, 'delta_acc1')} | {metric_cell(value, 'delta_mrr')} |")
    if siglip:
        lines += ['', '## Encoder replication: independently retrained SigLIP', '',
                  '| Intervention | Method | Acc@1 | MRR | TIG | Off-target drift |', '|---|---|---:|---:|---:|---:|']
        for field, result in siglip.items():
            for method in ['inverse_cosine', 'max_ngram_cosine', 'rank+locality+control', 'shuffled-image control']:
                value = result['methods'][method]
                lines.append(f"| {field} | {method} | {metric_cell(value, 'delta_acc1')} | {metric_cell(value, 'delta_mrr')} | {metric_cell(value, 'TIG')} | {metric_cell(value, 'off_target_drift')} |")
    if faithfulness:
        lines += ['', '## Absolute localization and omitted-phrase recoverability', '',
                  'Recoverability is the CLIP similarity drop for the target phrase after mean-color masking the method-selected region.', '',
                  '| Intervention | Method | Absolute localization Acc@1 | Localization MRR | Phrase-similarity drop |', '|---|---|---:|---:|---:|']
        for field, result in faithfulness.items():
            for method, recovery in result['recoverability_similarity_drop'].items():
                rec = f"{recovery['mean']:.4f} [{recovery['ci95'][0]:.4f}, {recovery['ci95'][1]:.4f}]"
                if method == 'ground_truth_target':
                    lines.append(f'| {field} | {method} | - | - | {rec} |')
                    continue
                loc = result['localization'][method]
                acc = f"{loc['accuracy_at_1']['mean']:.4f} [{loc['accuracy_at_1']['ci95'][0]:.4f}, {loc['accuracy_at_1']['ci95'][1]:.4f}]"
                mrr = f"{loc['mrr']['mean']:.4f} [{loc['mrr']['ci95'][0]:.4f}, {loc['mrr']['ci95'][1]:.4f}]"
                lines.append(f'| {field} | {method} | {acc} | {mrr} | {rec} |')
    lines += ['', '## Claim-to-evidence audit', '',
              '| Claim | Evidence | Status |', '|---|---|---|',
              f'| RQ1: learned region complementarity is measurable | Positive controlled gaps and visual-shuffle degradation, but learned Acc@1 `{learned:.4f}` does not beat inverse cosine `{inverse:.4f}` | Mixed / primary superiority claim not supported |',
              f"| RQ2: response is local under text intervention | Frozen transfer is complete; only {frozen['reviewed_valid_rows'] if frozen else 0}/100 diagnostic rows passed all semantic checks, with no human confirmation | Task validity weak; no confirmatory claim |",
              f"| RQ3: transfer across datasets/encoders/regions | {'SigLIP independently retrained encoder check completed; ' if siglip else ''}automatic patch-cluster sensitivity completed; Visual Genome and independent region proposals not completed | {'Partially tested' if siglip else 'Unverified'} |",
              '| Existing importance methods solve complementarity | Algorithm-adapted CCI and Grad-ECLIP metrics are reported on the same boxes | Adaptation evidence; authors’ repository execution not performed |', '',
              '## Failure taxonomy', '',
              '- Strong simple baseline: inverse crop-text cosine is stronger than the learned scorer on controlled deletion.',
              '- Natural-caption difficulty: independently retrained performance is lower for natural captions, which change more than the target entity.',
              '- Locality/strength tradeoff: rank-only training creates large TIG but also large off-target drift; locality losses reduce both.',
              '- Attribution mismatch: pooled-patch, CCI, and Grad-ECLIP importance need not encode “unmentioned visual content.”',
              '- Annotation validity: exact phrase/ID filtering cannot rule out synonymous or implicit target mention in natural captions.', '',
              '## Decision gate', '',
              ('The gate selects **finish the rigorous course comparison and repair or replace the task before further learned-method development**. '
               'The frozen ensemble does not beat inverse cosine on the full deletion benchmark, while fewer than half of the stratified semantic-review rows pass all checks. '
               'No learned variant, untouched-set confirmation, Visual Genome expansion, or CVPR experiment is authorized by this evidence. '
               'Human review remains desirable, but cannot rescue claims from the current labels without a revised task.') if frozen else
              'The decision gate remains pending frozen transfer and semantic review.', '',
              '## Compute ledger', '', '| Job | Partition | Elapsed (s) | GPU-hours | State |', '|---|---|---:|---:|---|']
    for row in compute:
        if 'error' in row:
            lines.append(f"| error | - | - | - | {row['error']} |")
        else:
            lines.append(f"| {row['job_id']} | {row['partition']} | {row['elapsed_seconds']} | {row['gpu_hours']:.4f} | {row['state']} |")
    lines += ['', '## Utilization monitoring', '', '| Run | Samples | Mean GPU util. | Max GPU util. | Nonzero samples |', '|---|---:|---:|---:|---:|']
    for name, value in utilization.items():
        mean = '-' if value['mean_gpu_utilization_percent'] is None else f"{value['mean_gpu_utilization_percent']:.1f}%"
        maximum = '-' if value['max_gpu_utilization_percent'] is None else f"{value['max_gpu_utilization_percent']}%"
        nonzero = '-' if value['nonzero_sample_fraction'] is None else f"{100 * value['nonzero_sample_fraction']:.1f}%"
        lines.append(f"| {name} | {value['samples']} | {mean} | {maximum} | {nonzero} |")
    lines += ['', f"Recorded allocation total at report generation: `{sum(row.get('gpu_hours', 0) for row in compute):.4f}` GPU-hours. This snapshot is not a final authorization balance; reconcile final cumulative usage under GOAL.md before submission. The ledger is a generation-time snapshot; final job state and cumulative usage must be recovered from sacct before new allocation.", '',
              '## Reproduction', '', '```sh',
              'python3 -u pilot/fetch_annotations.py',
              'python3 -u pilot/prepare.py --counts 200,50,100',
              'python3 -u pilot/audit.py',
              'sbatch hpc/full_l4.sbatch',
              'python3 -u pilot/next_round.py preflight --source SOURCE_TMINUS --output FOLLOWUP_DIR',
              'python3 -u pilot/visibility_audit.py --manifest SOURCE_TMINUS/manifest_snapshot.json --output FOLLOWUP_DIR/visibility_audit.json',
              'python3 -u pilot/next_round.py gpu --source SOURCE_TMINUS --output FOLLOWUP_DIR',
              'python3 -u pilot/render_semantic_audit.py --audit FOLLOWUP_DIR/semantic_audit.csv --output FOLLOWUP_DIR/audit-pages',
              'python3 -u pilot/next_round.py summarize --output FOLLOWUP_DIR',
              'python3 -u pilot/make_report.py --formal-root pilot/results/formal-l4-JOB_ID --job-ids JOB_ID',
              '```', '',
              'The dataset, model cache, raw predictions, and checkpoints are intentionally excluded from Git. Compact audited metrics and this report are committed.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines))
    print(args.report)
    print(args.compact)


if __name__ == '__main__':
    main()
