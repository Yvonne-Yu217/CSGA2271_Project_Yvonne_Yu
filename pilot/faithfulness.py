"""Absolute localization and omitted-phrase recoverability evaluation."""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageStat
from torch import nn
from transformers import CLIPModel, CLIPProcessor

MODEL = 'openai/clip-vit-base-patch16'


def scalar_summary(per_image, seed=2271):
    values = np.array(list(per_image.values()), dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([values[rng.integers(len(values), size=len(values))].mean() for _ in range(2000)])
    return {'mean': float(values.mean()), 'ci95': [float(x) for x in np.quantile(boot, [.025, .975])], 'images': len(values)}


def image_average(rows, values):
    grouped = collections.defaultdict(list)
    for row, value in zip(rows, values):
        grouped[row['image_id']].append(float(value))
    return {key: float(np.mean(items)) for key, items in grouped.items()}


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=root / 'data')
    ap.add_argument('--output-dir', type=Path, default=root / 'results' / 'faithfulness')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    started = time.time(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = json.loads((args.run_dir / 'manifest_snapshot.json').read_text())
    predictions = np.load(args.run_dir / 'predictions.npz')
    ptr = np.cumsum([0] + [len(row['regions']) for row in all_rows])
    row_indices = [i for i, row in enumerate(all_rows) if row['split'] == 'test']
    rows = [all_rows[i] for i in row_indices]
    rng = np.random.default_rng(2271)
    learned = np.mean([predictions[f'rank_local_control_seed{seed}'] for seed in [0, 1, 2]], axis=0)
    methods = {
        'learned_rank_local_control': learned,
        'inverse_cosine': predictions['inverse_cosine'],
        'max_ngram_cosine': predictions['max_ngram_cosine'],
        'region_area': predictions['region_area'],
        'random': rng.random(predictions['inverse_cosine'].shape),
    }
    selected = {name: [] for name in methods}
    localization = {}
    for name, scores in methods.items():
        ranks, hits = [], []
        for i in row_indices:
            a, b = ptr[i:i + 2]
            values = scores[a:b, 0]
            choice = int(np.argmax(values))
            selected[name].append(choice)
            hits.append(choice == 0)
            ranks.append(1 / (1 + int((values[1:] > values[0]).sum())))
        localization[name] = {
            'accuracy_at_1': scalar_summary(image_average(rows, hits)),
            'mrr': scalar_summary(image_average(rows, ranks)),
        }
    selected['ground_truth_target'] = [0] * len(rows)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable')
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    image_ids = sorted({row['image_id'] for row in rows})
    full_features = {}
    with torch.inference_mode():
        for batch_ids in chunks(image_ids, args.batch_size):
            images = []
            for ident in batch_ids:
                with Image.open(args.data_dir / 'images' / f'{ident}.jpg') as image:
                    images.append(image.convert('RGB'))
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            values = nn.functional.normalize(model.get_image_features(pixel_values=pixels), dim=-1).cpu().numpy()
            full_features.update(zip(batch_ids, values))
    keys = list(dict.fromkeys((row['image_id'], tuple(row['regions'][choice]['box']))
                              for name in selected for row, choice in zip(rows, selected[name])))
    masked_features = {}
    with torch.inference_mode():
        for batch_keys in chunks(keys, args.batch_size):
            images = []
            for ident, box in batch_keys:
                with Image.open(args.data_dir / 'images' / f'{ident}.jpg') as image:
                    image = image.convert('RGB')
                    fill = tuple(int(x) for x in ImageStat.Stat(image).mean)
                    array = np.array(image, copy=True)
                    x1, y1, x2, y2 = box
                    array[y1:y2, x1:x2] = fill
                    images.append(Image.fromarray(array))
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            values = nn.functional.normalize(model.get_image_features(pixel_values=pixels), dim=-1).cpu().numpy()
            masked_features.update(zip(batch_keys, values))
            print('masked', len(masked_features), '/', len(keys), flush=True)
    phrases = sorted({row['target_phrase'] for row in rows})
    phrase_features = {}
    with torch.inference_mode():
        for batch_phrases in chunks(phrases, args.batch_size):
            inputs = processor(text=['a photo of ' + value for value in batch_phrases], return_tensors='pt',
                               padding=True, truncation=True, max_length=77).to(device)
            values = nn.functional.normalize(model.get_text_features(**inputs), dim=-1).cpu().numpy()
            phrase_features.update(zip(batch_phrases, values))
    recoverability = {}
    raw = {}
    for name, choices in selected.items():
        drops = []
        for row, choice in zip(rows, choices):
            text = phrase_features[row['target_phrase']]
            full = full_features[row['image_id']]
            key = (row['image_id'], tuple(row['regions'][choice]['box']))
            drops.append(float(full @ text - masked_features[key] @ text))
        raw[name] = drops
        recoverability[name] = scalar_summary(image_average(rows, drops))
    report = {
        'scope': 'held-out absolute region selection and target-phrase recoverability after mean-color masking',
        'images': len(image_ids), 'pairs': len(rows), 'localization': localization,
        'recoverability_similarity_drop': recoverability,
        'runtime': {'seconds': time.time() - started, 'device': str(device),
                    'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None},
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    (args.output_dir / 'per_pair.json').write_text(json.dumps({'row_indices': row_indices, 'selected': selected, 'drops': raw}))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
