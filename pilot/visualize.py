"""Render qualitative complementarity maps and a machine-readable error audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=root / 'data')
    ap.add_argument('--output-dir', type=Path, default=root.parents[0] / 'research' / 'figures')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run_dir / 'manifest_snapshot.json').read_text())
    predictions = np.load(args.run_dir / 'predictions.npz')
    ensemble = np.mean([predictions[f'rank_local_control_seed{seed}'] for seed in [0, 1, 2]], axis=0)
    ptr = np.cumsum([0] + [len(row['regions']) for row in rows])
    records = []
    for i, row in enumerate(rows):
        if row['split'] != 'test':
            continue
        a, b = ptr[i:i + 2]
        delta = ensemble[a:b, 0] - ensemble[a:b, 1]
        rank = int(1 + (delta[1:] > delta[0]).sum())
        records.append({'row_index': i, 'rank': rank, 'target_gap': float(delta[0]),
                        'off_target_drift': float(np.abs(ensemble[a + 1:b, 0] - ensemble[a + 1:b, 1]).mean()),
                        'delta': delta.tolist()})
    success = sorted((r for r in records if r['rank'] == 1), key=lambda r: r['target_gap'], reverse=True)
    failure = sorted((r for r in records if r['rank'] > 1), key=lambda r: (r['rank'], -r['target_gap']), reverse=True)
    selected, used = [], set()
    for category, candidates in [('success', success), ('failure', failure)]:
        for record in candidates:
            ident = rows[record['row_index']]['image_id']
            if ident in used:
                continue
            record = dict(record, category=category)
            selected.append(record); used.add(ident)
            if sum(x['category'] == category for x in selected) == 3:
                break
    font = ImageFont.load_default()
    index = []
    for number, record in enumerate(selected, 1):
        row = rows[record['row_index']]
        with Image.open(args.data_dir / 'images' / f"{row['image_id']}.jpg") as source:
            source = source.convert('RGB')
        scale = min(1.0, 800 / source.width)
        if scale < 1:
            source = source.resize((round(source.width * scale), round(source.height * scale)))
        canvas = source.convert('RGBA')
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        values = np.array(record['delta'])
        lo, hi = float(values.min()), float(values.max())
        for region, value in zip(row['regions'], values):
            strength = .5 if hi == lo else (float(value) - lo) / (hi - lo)
            color = (round(255 * strength), 40, round(255 * (1 - strength)), 90)
            x1, y1, x2, y2 = [round(v * scale) for v in region['box']]
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=color[:3] + (255,), width=3)
            label = f"{region['phrase']}: {value:+.3f}"
            box = draw.textbbox((x1 + 2, y1 + 2), label, font=font)
            draw.rectangle(box, fill=(0, 0, 0, 190))
            draw.text((x1 + 2, y1 + 2), label, fill='white', font=font)
        canvas = Image.alpha_composite(canvas, overlay).convert('RGB')
        name = f"{number:02d}-{record['category']}-{row['image_id']}.jpg"
        canvas.save(args.output_dir / name, quality=85, optimize=True)
        index.append({**record, 'file': name, 'image_id': row['image_id'], 'target_phrase': row['target_phrase'],
                      'tminus': row['tminus'], 'tplus': row['tplus'], 'regions': row['regions']})
    taxonomy = {
        'test_pairs': len(records),
        'target_rank1': sum(r['rank'] == 1 for r in records),
        'target_not_rank1': sum(r['rank'] > 1 for r in records),
        'nonpositive_target_gap': sum(r['target_gap'] <= 0 for r in records),
        'off_target_drift_exceeds_abs_target_gap': sum(r['off_target_drift'] > abs(r['target_gap']) for r in records),
        'examples': index,
    }
    (args.output_dir / 'qualitative_index.json').write_text(json.dumps(taxonomy, indent=2))
    print(json.dumps({k: v for k, v in taxonomy.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
