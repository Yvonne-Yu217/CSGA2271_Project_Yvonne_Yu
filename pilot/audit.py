"""Audit a prepared Flickr30K Entities intervention manifest."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def contains_phrase(text, phrase):
    words, target = norm(text), norm(phrase)
    return any(words[i:i + len(target)] == target for i in range(len(words) - len(target) + 1))


def overlaps(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results' / 'data_audit.json')
    args = ap.parse_args()
    rows = json.loads((args.data_dir / 'manifest.json').read_text())
    provenance = json.loads((args.data_dir / 'provenance.json').read_text())
    errors, warnings = [], []
    split_images = collections.defaultdict(set)
    split_pairs = collections.Counter()
    region_counts = []
    captions = collections.defaultdict(set)
    seen = set()
    natural_counts = collections.Counter()
    for i, row in enumerate(rows):
        split = row.get('split')
        ident = row.get('image_id')
        split_images[split].add(ident)
        split_pairs[split] += 1
        captions[split].add(' '.join(norm(row.get('tplus', ''))))
        key = (ident, row.get('caption_index'), row.get('target_phrase'))
        if key in seen:
            errors.append(f'row {i}: duplicate intervention key {key}')
        seen.add(key)
        regions = row.get('regions', [])
        region_counts.append(len(regions))
        if len(regions) < 2:
            errors.append(f'row {i}: fewer than two regions')
            continue
        if not contains_phrase(row['tplus'], row['target_phrase']):
            errors.append(f'row {i}: target phrase absent from tplus')
        if contains_phrase(row['tminus'], row['target_phrase']):
            warnings.append(f'row {i}: target phrase still appears verbatim in tminus')
        if len(row.get('tmatched', '').split()) != len(row['tplus'].split()):
            errors.append(f'row {i}: matched intervention changed token count')
        if contains_phrase(row.get('tmatched', ''), row['target_phrase']):
            errors.append(f'row {i}: target phrase appears in tmatched')
        if row.get('tnatural'):
            natural_counts[split] += 1
            if contains_phrase(row['tnatural'], row['target_phrase']):
                errors.append(f'row {i}: target phrase appears in tnatural')
        if not contains_phrase(row['tcontrol'], row['target_phrase']):
            errors.append(f'row {i}: unrelated control removed target phrase')
        if row['tplus'] == row['tminus'] or row['tplus'] == row['tcontrol']:
            errors.append(f'row {i}: ineffective text intervention')
        target = regions[0]['box']
        for j, region in enumerate(regions):
            box = region['box']
            if len(box) != 4 or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
                errors.append(f'row {i} region {j}: invalid box {box}')
            if j and overlaps(target, box):
                errors.append(f'row {i} region {j}: overlaps target')
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = split_images[left] & split_images[right]
        if overlap:
            errors.append(f'image leakage {left}/{right}: {len(overlap)}')
        text_overlap = captions[left] & captions[right]
        if text_overlap:
            warnings.append(f'normalized caption overlap {left}/{right}: {len(text_overlap)}')
    hash_failures = []
    size_failures = []
    for ident, expected in provenance['image_sha256'].items():
        path = args.data_dir / 'images' / f'{ident}.jpg'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            hash_failures.append(ident)
            continue
        with Image.open(path) as image:
            width, height = image.size
        for row in (r for r in rows if r['image_id'] == ident):
            for region in row['regions']:
                x1, y1, x2, y2 = region['box']
                if x2 > width or y2 > height:
                    size_failures.append((ident, region['box'], [width, height]))
    if hash_failures:
        errors.append(f'image hash failures: {len(hash_failures)}')
    if size_failures:
        errors.append(f'boxes outside image: {len(size_failures)}')
    report = {
        'status': 'pass' if not errors else 'fail',
        'image_counts': {s: len(split_images[s]) for s in ['train', 'val', 'test']},
        'pair_counts': {s: split_pairs[s] for s in ['train', 'val', 'test']},
        'total_images': len(set().union(*split_images.values())),
        'total_pairs': len(rows),
        'natural_pair_counts': {s: natural_counts[s] for s in ['train', 'val', 'test']},
        'region_count': {
            'min': min(region_counts),
            'max': max(region_counts),
            'mean': sum(region_counts) / len(region_counts),
        },
        'verified_image_hashes': len(provenance['image_sha256']) - len(hash_failures),
        'errors': errors,
        'warnings': warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
