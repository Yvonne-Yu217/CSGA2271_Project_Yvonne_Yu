"""Stage new image-disjoint Flickr30K material for E0 human adjudication.

This command intentionally does *not* create gold facts or action labels.  It
downloads resumable source material, creates label-free grid candidates, and
exports annotation evidence into ``review_input`` for later two-person review.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import struct
import time
import zlib
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import fsspec
import requests
from PIL import Image

from pilot.prepare import BASE, IMAGE_URL, parse_caption


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def opaque(prefix, *parts):
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def dhash(path):
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def hamming_hex(left, right):
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def annotation_evidence(image_id, xml_bytes, caption_lines):
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes = collections.defaultdict(list)
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [int(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        coords = [max(0, coords[0] - 1), max(0, coords[1] - 1),
                  min(width, coords[2]), min(height, coords[3])]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        for name in obj.findall("name"):
            boxes[name.text].append(coords)
    captions, evidence = [], []
    for caption_index, line in enumerate(caption_lines):
        text, phrases = parse_caption(line)
        captions.append(text)
        for phrase_index, phrase in enumerate(phrases):
            phrase_boxes = boxes.get(phrase["id"], [])
            evidence.append({
                "review_item_id": opaque("rv", image_id, caption_index, phrase_index),
                "source_image_id": image_id, "caption_index": caption_index,
                "caption": text, "source_entity_id": phrase["id"], "phrase": phrase["phrase"],
                "source_types": phrase["types"], "boxes": phrase_boxes,
                "review_status": "unreviewed",
                "warning": "Source annotation evidence only; not E0 gold and not selector-visible.",
            })
    return width, height, captions, evidence


def grid_candidates(image_id, width, height):
    boxes = []
    for rows, columns in ((3, 3), (1, 2), (2, 1)):
        for row in range(rows):
            for column in range(columns):
                x1, x2 = round(column * width / columns), round((column + 1) * width / columns)
                y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
                boxes.append([x1, y1, x2, y2])
    boxes.append([0, 0, width, height])
    rows = []
    for index, box in enumerate(boxes):
        rows.append({
            "candidate_id": opaque("cand", image_id, index), "source_image_id": image_id,
            "kind": "bbox", "boxes": [box],
            "proposal_generator_revision": "fixed-grid-3x3-plus-strips-full-v1",
            "pixel_cost": (box[2] - box[0]) * (box[3] - box[1]),
        })
    rows.append({
        "candidate_id": opaque("cand", image_id, "stop"), "source_image_id": image_id,
        "kind": "stop", "boxes": [],
        "proposal_generator_revision": "fixed-grid-3x3-plus-strips-full-v1",
        "pixel_cost": 0,
    })
    return rows


class RemoteImageZip:
    def __init__(self):
        self.archive = zipfile.ZipFile(
            fsspec.open(IMAGE_URL, mode="rb", block_size=65536, cache_type="blockcache").open())
        infos = sorted(self.archive.infolist(), key=lambda item: item.header_offset)
        self.lookup = {Path(info.filename).stem: info for info in infos
                       if info.filename.startswith("flickr30k-images/") and info.filename.endswith(".jpg")}
        self.next_offset = {info.filename: infos[index + 1].header_offset
                            for index, info in enumerate(infos[:-1])}

    def fetch(self, image_id, destination):
        if destination.is_file():
            with Image.open(destination) as image:
                image.verify()
            return
        info = self.lookup[image_id]
        end = self.next_offset.get(info.filename)
        if end is None:
            end = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size
        response = None
        last_error = None
        for attempt in range(5):
            try:
                response = requests.get(
                    IMAGE_URL, headers={"Range": f"bytes={info.header_offset}-{end - 1}"},
                    timeout=90,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 4:
                    raise RuntimeError(f"failed to fetch {image_id} after 5 attempts") from exc
                time.sleep(min(2 ** attempt, 30))
        if response is None:
            raise RuntimeError(f"failed to fetch {image_id}: {last_error}")
        if response.status_code != 206:
            raise RuntimeError(f"image server ignored byte range for {image_id}")
        raw = response.content
        header = struct.unpack("<4s5H3I2H", raw[:30])
        if header[0] != b"PK\x03\x04":
            raise RuntimeError(f"bad local ZIP header for {image_id}")
        offset = 30 + header[-2] + header[-1]
        compressed = raw[offset:offset + info.compress_size]
        if info.compress_type == 8:
            payload = zlib.decompress(compressed, -15)
        elif info.compress_type == 0:
            payload = compressed
        else:
            raise RuntimeError(f"unsupported compression {info.compress_type}")
        if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
            raise RuntimeError(f"ZIP integrity failure for {image_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".partial")
        partial.write_bytes(payload)
        with Image.open(partial) as image:
            image.verify()
        partial.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-data", type=Path, default=Path("pilot/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--source-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--reuse-images-from", type=Path,
                        help="Optional earlier staging directory with identically named verified images")
    args = parser.parse_args()
    config = {"seed": args.seed, "requested_count": args.count,
              "source_split": args.source_split,
              "candidate_generator_revision": "fixed-grid-3x3-plus-strips-full-v1"}
    provenance_path = args.output / "provenance.json"
    if provenance_path.is_file():
        previous = json.loads(provenance_path.read_text())
        previous_config = {key: previous.get(key) for key in config}
        if previous_config != config:
            raise RuntimeError(f"output is pinned to another staging config: {previous_config}")
    args.output.mkdir(parents=True, exist_ok=True)
    journal_path = args.output / "progress.json"
    old_manifest = json.loads((args.pilot_data / "manifest.json").read_text())
    excluded_ids = {row["image_id"] for row in old_manifest}
    annotation_zip = zipfile.ZipFile(args.pilot_data / "annotations.zip")
    lookup = {Path(name).name: name for name in annotation_zip.namelist()
              if not name.startswith("__MACOSX")}
    split_path = args.pilot_data / f"{args.source_split}.txt"
    if not split_path.exists():
        response = requests.get(BASE + f"{args.source_split}.txt", timeout=60)
        response.raise_for_status()
        split_path.write_text(response.text)
    source_ids = split_path.read_text().split()
    random.Random(args.seed).shuffle(source_ids)
    selected = []
    for image_id in source_ids:
        if image_id in excluded_ids:
            continue
        xml_name, text_name = f"{image_id}.xml", f"{image_id}.txt"
        if xml_name not in lookup or text_name not in lookup:
            continue
        xml_bytes = annotation_zip.read(lookup[xml_name])
        caption_lines = annotation_zip.read(lookup[text_name]).decode().splitlines()
        width, height, captions, evidence = annotation_evidence(image_id, xml_bytes, caption_lines)
        if not evidence or not any(row["boxes"] for row in evidence):
            continue
        selected.append((image_id, width, height, captions, evidence))
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"only found {len(selected)} eligible images")

    remote = RemoteImageZip()
    old_dhashes = []
    for old_path in sorted((args.pilot_data / "images").glob("*.jpg")):
        old_dhashes.append((old_path.stem, dhash(old_path)))
    image_rows, context_rows, candidate_rows, review_rows = [], [], [], []
    warnings = []
    for index, (source_id, width, height, captions, evidence) in enumerate(selected, 1):
        image_id = opaque("img", source_id)
        image_path = args.output / "images" / f"{image_id}.jpg"
        if not image_path.exists() and args.reuse_images_from:
            reusable = args.reuse_images_from / "images" / image_path.name
            if reusable.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(reusable, image_path)
                except OSError:
                    shutil.copy2(reusable, image_path)
        remote.fetch(source_id, image_path)
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (width, height):
            raise RuntimeError(f"annotation/image size mismatch for {source_id}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        perceptual = dhash(image_path)
        close = [(old_id, hamming_hex(perceptual, old_hash)) for old_id, old_hash in old_dhashes
                 if hamming_hex(perceptual, old_hash) <= 4]
        if close:
            warnings.append({"source_image_id": source_id, "near_old_images": close,
                             "status": "requires_manual_duplicate_review"})
        image_rows.append({
            "staging_image_id": image_id, "source_image_id": source_id,
            "source_split": args.source_split, "image_path": str(image_path.relative_to(args.output)),
            "image_sha256": digest, "dhash64": perceptual, "width": width, "height": height,
            "old_pilot_excluded": True, "near_duplicate_review": "pending" if close else "clear",
        })
        for caption_index, caption in enumerate(captions):
            context_rows.append({
                "context_id": opaque("ctx", source_id, caption_index),
                "staging_image_id": image_id, "initial_caption": caption,
                "context_kind": "natural", "source_kind": "flickr30k-human-caption",
                "review_status": "unreviewed",
            })
        for row in grid_candidates(source_id, width, height):
            row["staging_image_id"] = image_id
            row.pop("source_image_id")
            candidate_rows.append(row)
        for row in evidence:
            row["staging_image_id"] = image_id
            review_rows.append(row)
        journal_path.write_text(json.dumps({
            "status": "running", "completed_images": index, "requested_images": args.count,
            "last_source_image_id": source_id,
        }, indent=2) + "\n")

    for left_index, left in enumerate(image_rows):
        for right in image_rows[left_index + 1:]:
            distance = hamming_hex(left["dhash64"], right["dhash64"])
            if distance <= 4:
                warnings.append({
                    "staging_image_ids": [left["staging_image_id"], right["staging_image_id"]],
                    "dhash_distance": distance,
                    "status": "requires_internal_duplicate_review",
                })

    write_jsonl(args.output / "staging_images.jsonl", image_rows)
    write_jsonl(args.output / "natural_contexts.jsonl", context_rows)
    write_jsonl(args.output / "automatic_candidates.jsonl", candidate_rows)
    write_jsonl(args.output / "review_input" / "source_entity_evidence.jsonl", review_rows)
    provenance = {
        "purpose": "E0 source staging; not adjudicated gold and not research evidence",
        "seed": args.seed, "requested_count": args.count, "source_split": args.source_split,
        "old_pilot_image_count_excluded": len(excluded_ids), "image_url": IMAGE_URL,
        "annotation_archive_sha256": hashlib.sha256(
            (args.pilot_data / "annotations.zip").read_bytes()).hexdigest(),
        "candidate_generator_revision": "fixed-grid-3x3-plus-strips-full-v1",
        "config_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = {
        "status": "staging_pass" if not warnings else "review_required",
        "e0_evaluated": False,
        "images": len(image_rows), "natural_contexts": len(context_rows),
        "candidates": len(candidate_rows), "source_entity_evidence_rows": len(review_rows),
        "old_pilot_overlap": sum(row["source_image_id"] in excluded_ids for row in image_rows),
        "near_duplicate_warnings": warnings,
        "limitations": [
            "Flickr30K Entities evidence is not an adjudicated E0 fact inventory.",
            "Natural captions are staged; enriched, paraphrase, and saturated states require review.",
            "Grid regions are an automatic smoke candidate set, not the final proposal generator.",
        ],
    }
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    journal_path.write_text(json.dumps({"status": "complete", **report}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
