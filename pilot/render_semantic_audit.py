"""Render semantic-audit CSV rows as review sheets with labeled boxes.

This is a presentation aid only. It never fills or changes audit decisions.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    names = [
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def wrapped(draw, position, text, width, face, fill, spacing=6):
    # Character wrapping is stable and sufficient for the English captions.
    lines = textwrap.wrap(text, width=width, break_long_words=False) or ['']
    draw.multiline_text(position, '\n'.join(lines), font=face, fill=fill, spacing=spacing)
    box = draw.multiline_textbbox(position, '\n'.join(lines), font=face, spacing=spacing)
    return box[3] - box[1]


def card(row, index, width=1800, height=560):
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title = font(28, bold=True)
    body = font(23)
    small = font(20)
    image = Image.open(row['image_path']).convert('RGB')
    scale = min(760 / image.width, 470 / image.height)
    shown = image.resize((round(image.width * scale), round(image.height * scale)))
    x0, y0 = 20, 70
    canvas.paste(shown, (x0, y0))
    target = json.loads(row['target_box'])
    def projected(box):
        return (x0 + round(box[0] * scale), y0 + round(box[1] * scale),
                x0 + round(box[2] * scale), y0 + round(box[3] * scale))
    draw.rectangle(projected(target),
                   outline='#e31a1c', width=6)
    for region in json.loads(row['other_regions']):
        box = region['box']
        draw.rectangle(projected(box),
                       outline='#33a02c', width=4)
        draw.text((x0 + round(box[0] * scale), y0 + round(box[1] * scale)),
                  region['phrase'], font=small, fill='#146b20', stroke_width=2, stroke_fill='white')
    draw.text((x0 + round(target[0] * scale), y0 + round(target[1] * scale)),
              row['target_phrase'], font=small, fill='#b30000', stroke_width=2, stroke_fill='white')
    draw.text((20, 18),
              f"#{index:03d}  {row['intervention']}  image={row['image_id']}  row={row['row_id']}",
              font=title, fill='black')
    tx, ty = 820, 75
    draw.text((tx, ty), f"TARGET (red): {row['target_phrase']}", font=title, fill='#b30000')
    ty += 52
    draw.text((tx, ty), 'ORIGINAL:', font=title, fill='black')
    ty += 42
    ty += wrapped(draw, (tx, ty), row['tplus'], 66, body, 'black') + 26
    draw.text((tx, ty), 'EDITED:', font=title, fill='black')
    ty += 42
    ty += wrapped(draw, (tx, ty), row['edited_caption'], 66, body, '#184a9c') + 26
    draw.text((tx, ty), 'CHECK: visible target; target omitted/generalized; controls unchanged; grammar.',
              font=small, fill='#555555')
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--per-page', type=int, default=4)
    args = ap.parse_args()
    with args.audit.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for start in range(0, len(rows), args.per_page):
        subset = rows[start:start + args.per_page]
        cards = [card(row, start + offset + 1) for offset, row in enumerate(subset)]
        page = Image.new('RGB', (cards[0].width, sum(item.height for item in cards)), '#d8d8d8')
        top = 0
        for item in cards:
            page.paste(item, (0, top))
            top += item.height
        name = f'page-{start // args.per_page + 1:03d}.jpg'
        page.save(args.output / name, quality=90)
        index.append({'page': name, 'first_row': start + 1, 'last_row': start + len(subset),
                      'row_ids': [row['row_id'] for row in subset]})
    (args.output / 'index.json').write_text(json.dumps(index, indent=2))
    print(f'Rendered {len(rows)} rows to {len(index)} pages in {args.output}')


if __name__ == '__main__':
    main()
