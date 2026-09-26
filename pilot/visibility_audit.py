"""CPU-only geometry audit of the historical CLIP spatial baseline's center crop.

Reports the continuous geometry used by baselines.patch_mask, not pixel-exact
processor rounding or the recall of an independent region proposal generator.
"""
import argparse
import collections
import json
from pathlib import Path


def geometry(box, size, crop=224, grid=14):
    width, height = size
    scale = crop / min(width, height)
    left, top = (width * scale - crop) / 2, (height * scale - crop) / 2
    x1, y1, x2, y2 = box
    x1, x2 = x1 * scale - left, x2 * scale - left
    y1, y2 = y1 * scale - top, y2 * scale - top
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        raise ValueError('Invalid region area')
    visible = max(0, min(crop, x2) - max(0, x1)) * max(0, min(crop, y2) - max(0, y1)) / area
    centers = [(i + .5) * crop / grid for i in range(grid)]
    patches = sum(x1 <= x < x2 and y1 <= y < y2 for x in centers for y in centers)
    return {'visible_fraction': visible, 'visible_patch_centers': patches,
            'historical_nearest_patch_fallback': patches == 0,
            'visibility_group': 'invisible' if visible == 0 else 'partial' if visible < 1 - 1e-8 else 'full'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    rows = json.loads(args.manifest.read_text())
    sizes, regions, pairs = {}, {}, []
    for row in rows:
        if row['split'] != 'test':
            continue
        ident = row['image_id']
        if ident not in sizes:
            with Image.open(args.data_dir / 'images' / (ident + '.jpg')) as image:
                sizes[ident] = image.size
        keys = []
        for region in row['regions']:
            key = json.dumps([ident, region['box']], separators=(',', ':'))
            keys.append(key)
            regions[key] = dict(image_id=ident, box=region['box'], **geometry(region['box'], sizes[ident]))
        pairs.append({'image_id': ident, 'caption_index': row['caption_index'],
                      'target_id': row['regions'][0]['id'], 'target_geometry': regions[keys[0]],
                      'all_candidates_fully_visible': all(regions[k]['visibility_group'] == 'full' for k in keys)})
    result = {'scope': __doc__, 'images': len(sizes), 'unique_regions': len(regions),
              'region_visibility_counts': dict(collections.Counter(r['visibility_group'] for r in regions.values())),
              'nearest_patch_fallback_regions': sum(r['historical_nearest_patch_fallback'] for r in regions.values()),
              'target_pair_visibility_counts': dict(collections.Counter(r['target_geometry']['visibility_group'] for r in pairs)),
              'regions': list(regions.values()), 'pairs': pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Visibility audit saved: {args.output}; {len(sizes)} images, {len(regions)} unique regions')


if __name__ == '__main__':
    main()
