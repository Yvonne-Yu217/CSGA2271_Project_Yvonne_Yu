"""Bounded frozen-checkpoint follow-up: CPU preflight, GPU inference, CPU summary.

This does not train a model or treat unreviewed natural captions as valid causal edits.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import time
from pathlib import Path

FIELDS = ('tminus', 'tmatched', 'tnatural')
SEEDS = (0, 1, 2)


def is_human_reviewer(name):
    normalized = name.strip().lower()
    nonhuman_markers = ('codex', 'nonhuman', 'non-human', 'model review', 'ai review')
    return bool(normalized) and not any(marker in normalized for marker in nonhuman_markers)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_id(row, field):
    value = [row['image_id'], row['caption_index'], row['regions'][0]['id'], field,
             row[field], row['tplus'], row['regions']]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def identity(args):
    required = ['manifest_snapshot.json', 'feature_runtime.json', 'metrics.json']
    required += [f'checkpoints/rank_local_control_seed{s}.pt' for s in SEEDS]
    for filename in required:
        if not (args.source / filename).is_file():
            raise ValueError(f'Missing source artifact: {args.source / filename}')
    runtime = read(args.source / 'feature_runtime.json')
    if runtime.get('model') != args.model:
        raise ValueError('Requested model differs from source feature_runtime.json; cross-encoder weights are forbidden')
    if read(args.source / 'metrics.json').get('intervention') != 'tminus':
        raise ValueError('Source must be the original deletion-trained tminus run')
    rows = read(args.source / 'manifest_snapshot.json')
    split_ids = {s: {r['image_id'] for r in rows if r['split'] == s}
                 for s in ('train', 'val', 'test')}
    if any(split_ids[a] & split_ids[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Source image splits overlap')
    test = [r for r in rows if r['split'] == 'test']
    if not test:
        raise ValueError('No held-out source rows')
    images = {}
    for ident in sorted({r['image_id'] for r in test}):
        path = args.data_dir / 'images' / f'{ident}.jpg'
        if not path.is_file():
            raise ValueError(f'Missing staged image: {path}')
        images[ident] = sha(path)
    for field in FIELDS:
        ids = [row_id(r, field) for r in test if r.get(field)]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Missing or duplicate evaluation rows for {field}')
    payload = {'source': str(args.source.resolve()), 'model': args.model,
               'source_files': {f: sha(args.source / f) for f in required},
               'image_sha256': images,
               'implementation_sha256': {name: sha(Path(__file__).with_name(name))
                                         for name in ('next_round.py', 'run.py')}}
    return payload, test


def preflight(args):
    payload, rows = identity(args)
    # Resolve only existing cache entries; never download while reserving a GPU.
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor
    snapshot = Path(snapshot_download(args.model, local_files_only=True))
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, use_fast=False)
    if not list(snapshot.glob('*.safetensors')) and not list(snapshot.glob('pytorch_model*.bin')):
        raise ValueError(f'No model weights staged at {snapshot}')
    # Validate dimensions and state dictionaries on CPU before submitting Slurm.
    import torch
    from run import Scorer
    dimensions = []
    for seed in SEEDS:
        state = torch.load(args.source / f'checkpoints/rank_local_control_seed{seed}.pt',
                           map_location='cpu', weights_only=True)
        dimension = state['net.0.weight'].shape[1] // 4
        Scorer('rank_local_control', dimension).load_state_dict(state, strict=True)
        dimensions.append(dimension)
    if len(set(dimensions)) != 1:
        raise ValueError('Checkpoint feature dimensions differ')
    projection_dim = getattr(config, 'projection_dim', None)
    if projection_dim is None and getattr(config, 'model_type', None) == 'siglip':
        projection_dim = config.text_config.hidden_size
        if projection_dim != config.vision_config.hidden_size:
            raise ValueError('Unsupported SigLIP text/vision projection configuration')
    if projection_dim is None or projection_dim != dimensions[0]:
        raise ValueError(f'Pinned encoder projection dimension {projection_dim} differs from checkpoint dimension {dimensions[0]}')
    payload['feature_dim'] = dimensions[0]
    payload['encoder_projection_dim'] = projection_dim
    payload['model_snapshot'] = str(snapshot.resolve())
    payload['model_files'] = {p.name: sha(p) for p in snapshot.iterdir()
                              if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin', '.txt', '.model'}}
    payload['limitations'] = ['Legacy source metadata records model ID, not exact training-time model revision; '
                             'the current local snapshot is pinned and hashed for this follow-up.']
    plan_path = args.output / 'preflight.json'
    if plan_path.exists() and read(plan_path) != payload:
        raise ValueError('Existing preflight differs; use a fresh output directory')
    write(plan_path, payload)
    audit = args.output / 'semantic_audit.csv'
    if not audit.exists():
        bins = collections.defaultdict(list)
        for field in FIELDS:
            for row in rows:
                if row.get(field):
                    bins[(field, min(len(row['regions']), 4))].append(row)
        rng = random.Random(2271)
        for group in bins.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < args.audit_size and any(bins.values()):
            for (field, _), group in sorted(bins.items()):
                if group and len(selected) < args.audit_size:
                    selected.append((field, group.pop()))
        headers = ['row_id', 'intervention', 'image_id', 'image_path', 'target_phrase',
                   'target_box', 'other_regions', 'tplus', 'edited_caption', 'target_change_valid',
                   'controls_preserved', 'grammar_acceptable', 'audit_decision', 'reviewer', 'notes']
        with audit.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            for field, row in selected:
                writer.writerow(dict(row_id=row_id(row, field), intervention=field, image_id=row['image_id'],
                    image_path=str((args.data_dir / 'images' / (row['image_id'] + '.jpg')).resolve()),
                    target_phrase=row['target_phrase'], target_box=json.dumps(row['regions'][0]['box']),
                    other_regions=json.dumps(row['regions'][1:]), tplus=row['tplus'], edited_caption=row[field],
                    target_change_valid='', controls_preserved='', grammar_acceptable='',
                    audit_decision='unreviewed', reviewer='', notes=''))
    print(f'Preflight passed; {len(rows)} held-out pairs; audit template: {audit}', flush=True)


def gpu(args):
    import numpy as np
    import torch
    import run
    if not torch.cuda.is_available():
        raise ValueError('GPU stage requires CUDA')
    payload, rows = identity(args)
    prepared = read(args.output / 'preflight.json')
    if any(prepared.get(k) != v for k, v in payload.items()):
        raise ValueError('Inputs changed after CPU preflight; restage before allocating a GPU')
    snapshot = Path(prepared['model_snapshot'])
    for name, digest in prepared['model_files'].items():
        if not (snapshot / name).is_file() or sha(snapshot / name) != digest:
            raise ValueError('Pinned model cache changed after preflight')
    run.DATA = args.data_dir
    run.MODEL = str(snapshot)
    run.DEVICE = 'cuda'
    run.LOCAL_ONLY = True
    run.BATCH_SIZE = args.batch_size
    started = time.time()
    fingerprint = hashlib.sha256(json.dumps({'schema': 'shared_interventions_v1', 'inputs': prepared}, sort_keys=True).encode()).hexdigest()
    combined, field_rows, row_ranges = [], {}, {}
    for field in FIELDS:
        selected = [r for r in rows if isinstance(r.get(field), str) and r[field].strip()]
        field_rows[field] = selected
        row_ranges[field] = (len(combined), len(combined) + len(selected))
        # Copy each row: source manifests and original intervention strings remain unchanged.
        combined.extend(dict(row, tminus=row[field]) for row in selected)
    shared = args.output / 'shared_features'
    shared.mkdir(parents=True, exist_ok=True)
    cache_id = shared / 'cache_identity.json'
    if (shared / 'features.npz').exists() and (not cache_id.exists() or read(cache_id) != fingerprint):
        raise ValueError('Unverified existing cache; use a fresh output directory')
    write(cache_id, fingerprint)
    run.OUT = shared
    run.MINUS_FIELD = 'tminus'
    print(f'Extracting all interventions together: {len(combined)} pairs, batch={args.batch_size}', flush=True)
    cache = run.features(combined)
    visual_features, text_features = cache['v'], cache['t']
    if visual_features.shape[1] != prepared['feature_dim'] or text_features.shape[1] != prepared['feature_dim']:
        raise ValueError('Encoder features are incompatible with source checkpoints')
    ix, ptr = cache['index'], cache['ptr']
    all_predictions = {'inverse_cosine': np.stack([
        .5 * (1 - (visual_features[ix[:, 0]] * text_features[ix[:, j]]).sum(-1))
        for j in (1, 2, 3)], axis=1), 'max_ngram_cosine': cache['ngram_scores']}
    all_predictions['constant'] = np.zeros_like(all_predictions['inverse_cosine'])
    for seed in SEEDS:
        model = run.Scorer('rank_local_control', prepared['feature_dim']).cuda().eval()
        state = torch.load(args.source / f'checkpoints/rank_local_control_seed{seed}.pt',
                           map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        columns = []
        with torch.inference_mode():
            for column in (1, 2, 3):
                pieces = []
                for start in range(0, len(ix), args.batch_size * 8):
                    index = ix[start:start + args.batch_size * 8]
                    visual = torch.from_numpy(visual_features[index[:, 0]]).cuda()
                    text = torch.from_numpy(text_features[index[:, column]]).cuda()
                    pieces.append(model(visual, text).cpu().numpy())
                columns.append(np.concatenate(pieces))
        all_predictions[f'seed{seed}'] = np.stack(columns, axis=1)
        del model
        print(f'Frozen seed {seed}: all intervention predictions ready', flush=True)
    all_predictions['prediction_ensemble'] = np.mean([all_predictions[f'seed{s}'] for s in SEEDS], axis=0)
    for field, selected in field_rows.items():
        directory = args.output / field
        directory.mkdir(parents=True, exist_ok=True)
        first_row, end_row = row_ranges[field]
        first_flat, end_flat = int(ptr[first_row]), int(ptr[end_row])
        predictions = {name: values[first_flat:end_flat] for name, values in all_predictions.items()}
        np.savez_compressed(directory / 'predictions.npz', **predictions)
        write(directory / 'manifest_snapshot.json', selected)
        write(directory / 'inference.json', {'model': args.model, 'source': str(args.source.resolve()),
              'source_intervention': 'tminus', 'evaluation_intervention': field, 'training': False,
              'feature_identity': fingerprint, 'gpu': torch.cuda.get_device_name(),
              'shared_cache': str(shared.resolve()), 'shared_row_range': [first_row, end_row],
              'shared_flat_range': [first_flat, end_flat],
              'cumulative_seconds': time.time() - started})
        print(f'{field}: predictions saved for {len(selected)} pairs; no training performed', flush=True)
    cache.close()
    print('GPU work complete. Release allocation; run summarize on CPU.', flush=True)


def rank_metrics(values, np):
    target = values[0]
    higher = int((values > target + 1e-8).sum())
    equal = int((np.abs(values - target) <= 1e-8).sum())
    return (1 / equal if higher == 0 else 0.,
            float(np.mean([1 / (higher + index + 1) for index in range(equal)])))


METRICS = ['delta_acc1', 'delta_mrr', 'absolute_acc1', 'absolute_mrr', 'target_delta',
           'off_target_absolute_delta', 'normalized_target_delta', 'normalized_local_contrast']


def per_image(rows, scores, np):
    grouped = collections.defaultdict(list)
    offset = 0
    for row in rows:
        q = scores[offset:offset + len(row['regions'])]
        offset += len(row['regions'])
        delta = q[:, 0] - q[:, 1]
        scale = float(np.mean(np.abs(delta)))
        normalized = delta / scale if scale > 1e-8 else np.zeros_like(delta)
        values = [*rank_metrics(delta, np), *rank_metrics(q[:, 0], np), float(delta[0]),
                  float(np.abs(delta[1:]).mean()), float(normalized[0]),
                  float(normalized[0] - normalized[1:].mean())]
        grouped[row['image_id']].append(values)
    return {key: np.mean(value, axis=0) for key, value in grouped.items()}


def summarize_values(values, np):
    if not values:
        return {'images': 0, 'status': 'no eligible audited rows'}
    array = np.stack([values[key] for key in sorted(values)])
    rng = np.random.default_rng(2271)
    boot = np.stack([array[rng.integers(len(array), size=len(array))].mean(0) for _ in range(2000)])
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    return {'images': len(array), 'metrics': {name: {'mean': float(value), 'ci95': [float(lo), float(hi)]}
            for name, value, lo, hi in zip(METRICS, array.mean(0), lower, upper)}}


def summary(args):
    import numpy as np
    with (args.output / 'semantic_audit.csv').open(newline='') as stream:
        audit = list(csv.DictReader(stream))
    valid = {r['row_id'] for r in audit if r['audit_decision'].strip().lower() == 'valid'
             and all(r[k].strip().lower() == 'yes' for k in
                     ('target_change_valid', 'controls_preserved', 'grammar_acceptable'))
             and r['reviewer'].strip()}
    human_valid = {r['row_id'] for r in audit if r['row_id'] in valid
                   and is_human_reviewer(r['reviewer'])}
    report = {'scope': 'Frozen deletion-trained checkpoint follow-up; no retraining',
              'audit_counts': dict(collections.Counter(r['audit_decision'] for r in audit)),
              'audit_counts_by_intervention': {field: dict(collections.Counter(
                  r['audit_decision'] for r in audit if r['intervention'] == field)) for field in FIELDS},
              'reviewer_counts': dict(collections.Counter(r['reviewer'] or 'unreviewed' for r in audit)),
              'reviewed_valid_rows': len(valid),
              'human_confirmed_valid_rows': len(human_valid),
              'limitations': ['Unreviewed natural-caption delta/locality scores are exploratory because multiple entities may change.',
                 'Reviewer identity is explicit; model visual review is diagnostic and is never labeled human confirmation.',
                 'Image bootstrap conditions on the three existing seeds; it is not a training-seed uncertainty interval.',
                 'Normalized deltas divide by each pair mean absolute delta; values with denominator <= 1e-8 map to zero.',
                 'Audited subsets are small stratified diagnostic samples, not unbiased population estimates.'],
              'interventions': {}}
    all_rows = {f: read(args.output / f / 'manifest_snapshot.json') for f in FIELDS}
    common = set.intersection(*({(r['image_id'], r['caption_index'], r['regions'][0]['id']) for r in rows}
                                for rows in all_rows.values()))
    for field, rows in all_rows.items():
        with np.load(args.output / field / 'predictions.npz') as archive:
            scores = {key: archive[key] for key in archive.files}
        ptr = np.cumsum([0] + [len(r['regions']) for r in rows])
        result = {}
        subsets = ['all_exploratory', 'common_pairs', 'audited_valid']
        subsets += [f'candidate_count_{count}' for count in sorted({len(r['regions']) for r in rows})]
        for subset in subsets:
            indices = [i for i, r in enumerate(rows) if subset == 'all_exploratory'
                       or (subset == 'common_pairs' and (r['image_id'], r['caption_index'], r['regions'][0]['id']) in common)
                       or (subset == 'audited_valid' and row_id(r, field) in valid)
                       or (subset.startswith('candidate_count_') and len(r['regions']) == int(subset.rsplit('_', 1)[1]))]
            selected = [rows[i] for i in indices]
            flat = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in indices]) if indices else np.array([], dtype=int)
            per = {key: per_image(selected, value[flat], np) for key, value in scores.items()}
            per['seed_metric_mean'] = {image: np.mean([per[f'seed{s}'][image] for s in SEEDS], axis=0)
                                       for image in per['seed0']}
            result[subset] = {'pairs': len(selected), 'methods': {key: summarize_values(value, np) for key, value in per.items()},
                'paired_vs_inverse_cosine': {key: summarize_values({image: value[image] - per['inverse_cosine'][image]
                    for image in value}, np) for key, value in per.items() if key != 'inverse_cosine'}}
        report['interventions'][field] = result
    write(args.output / 'summary.json', report)
    print(f'CPU summary saved: {args.output / "summary.json"}; reviewed valid rows: {len(valid)}; '
          f'human-confirmed valid rows: {len(human_valid)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'gpu', 'summarize'])
    parser.add_argument('--source', type=Path, help='Existing deletion run containing checkpoints and provenance')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--audit-size', type=int, default=100)
    args = parser.parse_args()
    if args.stage != 'summarize' and not args.source:
        parser.error('--source is required for preflight and gpu')
    if args.batch_size < 1 or args.audit_size < 1:
        parser.error('batch-size and audit-size must be positive')
    {'preflight': preflight, 'gpu': gpu, 'summarize': summary}[args.stage](args)


if __name__ == '__main__':
    main()
