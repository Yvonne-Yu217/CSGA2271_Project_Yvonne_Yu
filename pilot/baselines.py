"""Held-out spatial baselines for the entity intervention benchmark.

The decomposition, masking, and gradient methods here are task-adapted CLIP
proxies evaluated on the same ground-truth boxes. They are not claimed to be
the authors' official Grad-ECLIP or CCI implementations.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageStat
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from run import per_image, summarize

MODEL = 'openai/clip-vit-base-patch16'


def patch_mask(box, size, grid=14, crop=224):
    width, height = size
    scale = crop / min(width, height)
    left = (width * scale - crop) / 2
    top = (height * scale - crop) / 2
    x1, y1, x2, y2 = box
    transformed = (x1 * scale - left, y1 * scale - top, x2 * scale - left, y2 * scale - top)
    centers = (np.arange(grid) + .5) * crop / grid
    yy, xx = np.meshgrid(centers, centers, indexing='ij')
    mask = (xx >= transformed[0]) & (xx < transformed[2]) & (yy >= transformed[1]) & (yy < transformed[3])
    if not mask.any():
        # Use the closest visible patch for tiny/clipped regions.
        cx = np.clip((transformed[0] + transformed[2]) / 2, 0, crop)
        cy = np.clip((transformed[1] + transformed[3]) / 2, 0, crop)
        dist = (xx - cx) ** 2 + (yy - cy) ** 2
        mask.flat[int(dist.argmin())] = True
    return mask.reshape(-1)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield start, values[start:start + size]


def cosine_kmeans(features, clusters=5, iterations=20):
    """Small deterministic cosine k-means for one image's patch tokens."""
    values = nn.functional.normalize(features.float(), dim=-1)
    seeds = [0]
    for _ in range(1, min(clusters, len(values))):
        similarity = values @ values[seeds].T
        seeds.append(int(similarity.max(dim=1).values.argmin()))
    centers = values[seeds]
    labels = None
    for _ in range(iterations):
        new_labels = (values @ centers.T).argmax(dim=1)
        if labels is not None and torch.equal(new_labels, labels):
            break
        labels = new_labels
        updated = []
        for index in range(len(centers)):
            members = values[labels == index]
            updated.append(nn.functional.normalize(members.mean(0), dim=0) if len(members) else centers[index])
        centers = torch.stack(updated)
    return labels


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--data-dir', type=Path, default=root / 'data')
    ap.add_argument('--output-dir', type=Path, default=root / 'results' / 'spatial-baselines')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--split', default='test')
    ap.add_argument('--minus-field', choices=['tminus', 'tmatched', 'tnatural'], default='tminus')
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--clusters', type=int, default=5,
                    help='Number of cosine patch clusters for the CCI adaptation')
    args = ap.parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((args.data_dir / 'manifest.json').read_text())
            if r['split'] == args.split and isinstance(r.get(args.minus_field), str)]
    if args.max_images:
        image_ids = list(dict.fromkeys(r['image_id'] for r in rows))[:args.max_images]
        rows = [r for r in rows if r['image_id'] in set(image_ids)]
    if not rows:
        ap.error('No rows selected')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable')
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    text_keys = [args.minus_field, 'tplus', 'tcontrol']
    texts = sorted({r[k] for r in rows for k in text_keys})
    text_index = {text: i for i, text in enumerate(texts)}
    text_features = []
    with torch.inference_mode():
        for _, batch_texts in chunks(texts, args.batch_size):
            inputs = processor(text=batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
            text_features.append(nn.functional.normalize(model.get_text_features(**inputs), dim=-1).cpu())
    text_features = torch.cat(text_features).numpy()

    image_ids = sorted({r['image_id'] for r in rows})
    image_features = {}
    patch_features = {}
    image_sizes = {}
    cci_labels = {}
    cci_masked_features = {}
    with torch.inference_mode():
        for _, batch_ids in chunks(image_ids, args.batch_size):
            images = []
            for ident in batch_ids:
                with Image.open(args.data_dir / 'images' / f'{ident}.jpg') as image:
                    image_sizes[ident] = image.size
                    images.append(image.convert('RGB'))
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            vision = model.vision_model(pixel_values=pixels, output_hidden_states=True)
            full = nn.functional.normalize(model.visual_projection(vision.pooler_output), dim=-1)
            patches = model.vision_model.post_layernorm(vision.last_hidden_state[:, 1:])
            patches = nn.functional.normalize(model.visual_projection(patches), dim=-1)
            for ident, full_feature, patch_feature in zip(batch_ids, full.cpu().numpy(), patches.cpu().numpy()):
                image_features[ident] = full_feature
                patch_features[ident] = patch_feature
            # CCI adaptation: cluster penultimate patch tokens, then suppress
            # CLS-to-cluster attention only in the final transformer block.
            penultimate = vision.hidden_states[-2]
            final_layer = model.vision_model.encoder.layers[-1]
            for index, ident in enumerate(batch_ids):
                labels = cosine_kmeans(penultimate[index, 1:], clusters=args.clusters)
                count = int(labels.max()) + 1
                repeated = penultimate[index:index + 1].repeat(count, 1, 1)
                attention_mask = torch.zeros(count, 1, repeated.shape[1], repeated.shape[1],
                                             device=device, dtype=repeated.dtype)
                for cluster in range(count):
                    attention_mask[cluster, 0, 0, 1:][labels == cluster] = torch.finfo(repeated.dtype).min
                masked_hidden = final_layer(repeated, attention_mask=attention_mask,
                                            causal_attention_mask=None, output_attentions=False)[0]
                masked_cls = model.vision_model.post_layernorm(masked_hidden[:, 0])
                masked_cls = nn.functional.normalize(model.visual_projection(masked_cls), dim=-1)
                cci_labels[ident] = labels.cpu().numpy()
                cci_masked_features[ident] = masked_cls.cpu().numpy()

    region_keys = list(dict.fromkeys((r['image_id'], tuple(region['box'])) for r in rows for region in r['regions']))
    masked_features = {}
    with torch.inference_mode():
        for _, batch_keys in chunks(region_keys, args.batch_size):
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

    ptr = [0]
    for row in rows:
        ptr.append(ptr[-1] + len(row['regions']))
    ptr = np.array(ptr)
    total_regions = ptr[-1]
    patch_scores = np.zeros((total_regions, 3), dtype=np.float32)
    masking_scores = np.zeros_like(patch_scores)
    cci_scores = np.zeros_like(patch_scores)
    for i, row in enumerate(rows):
        text = text_features[[text_index[row[k]] for k in text_keys]]
        full = image_features[row['image_id']]
        for j, region in enumerate(row['regions']):
            flat = ptr[i] + j
            mask = patch_mask(region['box'], image_sizes[row['image_id']])
            pooled = patch_features[row['image_id']][mask].mean(0)
            pooled /= max(np.linalg.norm(pooled), 1e-12)
            patch_scores[flat] = .5 * (1 - pooled @ text.T)
            masked = masked_features[(row['image_id'], tuple(region['box']))]
            # Negative importance: a region is less complementary when masking it
            # harms similarity to the current caption more strongly.
            masking_scores[flat] = masked @ text.T - full @ text.T
            cluster_importance = full @ text.T - cci_masked_features[row['image_id']] @ text.T
            labels = cci_labels[row['image_id']][mask]
            cci_scores[flat] = (-cluster_importance[labels]).mean(0)

    grad_scores = np.zeros_like(patch_scores)
    for column, key in enumerate(text_keys):
        for start, batch_rows in chunks(rows, args.batch_size):
            images, batch_texts = [], []
            for row in batch_rows:
                with Image.open(args.data_dir / 'images' / f"{row['image_id']}.jpg") as image:
                    images.append(image.convert('RGB'))
                batch_texts.append(row[key])
            inputs = processor(text=batch_texts, images=images, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
            pixels = inputs.pop('pixel_values').requires_grad_(True)
            text_inputs = {k: v for k, v in inputs.items() if k in {'input_ids', 'attention_mask'}}
            with torch.no_grad():
                text_batch = nn.functional.normalize(model.get_text_features(**text_inputs), dim=-1)
            cache = {}
            def capture_attention(module, inputs, kwargs, output):
                cache['hidden'] = kwargs['hidden_states']
                cache['output'] = output[0]
            handle = model.vision_model.encoder.layers[-1].self_attn.register_forward_hook(capture_attention, with_kwargs=True)
            vision = model.vision_model(pixel_values=pixels, output_hidden_states=False)
            handle.remove()
            image_batch = nn.functional.normalize(model.visual_projection(vision.pooler_output), dim=-1)
            paired_similarity = (image_batch * text_batch).sum(-1).sum()
            gradient = torch.autograd.grad(paired_similarity, cache['output'])[0][:, 0]
            attention = model.vision_model.encoder.layers[-1].self_attn
            q = attention.out_proj(attention.q_proj(cache['hidden']))[:, 0]
            k = attention.out_proj(attention.k_proj(cache['hidden']))[:, 1:]
            v = attention.out_proj(attention.v_proj(cache['hidden']))[:, 1:]
            q = nn.functional.normalize(q, dim=-1)
            k = nn.functional.normalize(k, dim=-1)
            key_similarity = (q[:, None] * k).sum(-1)
            key_similarity = (key_similarity - key_similarity.min(1, keepdim=True).values) / \
                             (key_similarity.max(1, keepdim=True).values - key_similarity.min(1, keepdim=True).values).clamp_min(1e-12)
            saliency = torch.relu((gradient[:, None] * v * key_similarity[..., None]).sum(-1)).detach().cpu().numpy()
            for offset, row in enumerate(batch_rows):
                i = start + offset
                for j, region in enumerate(row['regions']):
                    mask = patch_mask(region['box'], image_sizes[row['image_id']])
                    grad_scores[ptr[i] + j, column] = -float(saliency[offset][mask].mean())
            print('gradient', key, min(start + len(batch_rows), len(rows)), '/', len(rows), flush=True)

    methods = {
        'pooled_patch_inverse_cosine': patch_scores,
        'region_masking_inverse_importance': masking_scores,
        'cci_cluster_inverse_importance': cci_scores,
        'grad_eclip_inverse_saliency': grad_scores,
    }
    per = {name: per_image(rows, ptr, scores, range(len(rows))) for name, scores in methods.items()}
    report = {
        'scope': f'{args.split}-only task-adapted spatial baselines on ground-truth regions',
        'method_status': {
            'pooled_patch_inverse_cosine': 'crop-versus-pooled-patch ablation',
            'region_masking_inverse_importance': 'CCI-style masking proxy, not official CCI reproduction',
            'cci_cluster_inverse_importance': 'CCI algorithm adaptation: cosine patch clusters and final-block CLS attention masking',
            'grad_eclip_inverse_saliency': 'Grad-ECLIP formula adapted to the Hugging Face CLIP final vision-attention block',
        },
        'intervention': args.minus_field,
        'cci_clusters': args.clusters,
        'images': len({r['image_id'] for r in rows}),
        'pairs': len(rows),
        'metrics': {name: summarize(values) for name, values in per.items()},
        'runtime': {
            'seconds': time.time() - start_time,
            'device': str(device),
            'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
            'python': platform.python_version(),
        },
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    np.savez(args.output_dir / 'predictions.npz', **methods)
    print(json.dumps({k: v['delta_acc1'] for k, v in report['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
