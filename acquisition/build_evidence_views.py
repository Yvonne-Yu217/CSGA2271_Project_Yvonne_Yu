"""Build deterministic entity, padded, relation-union, full and STOP views."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from acquisition.observe import atomic_jsonl, read_jsonl


PADDING_FRACTION = 0.15
MAX_RELATION_UNIONS = 2
REVISION = "grounded-evidence-views-v1"


def opaque(prefix, *parts):
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def padded(box, width, height):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * PADDING_FRACTION, (y2 - y1) * PADDING_FRACTION
    return [max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy))),
            min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))]


def union(left, right):
    return [min(left[0], right[0]), min(left[1], right[1]),
            max(left[2], right[2]), max(left[3], right[3])]


def center_distance(left, right, width, height):
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    return math.hypot((lx - rx) / width, (ly - ry) / height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--grounding-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    grounding_path = args.grounding_output / "grounded_regions.jsonl"
    grounding = {row["staging_image_id"]: row for row in read_jsonl(grounding_path)}
    if set(grounding) != set(images):
        raise RuntimeError("grounding output must cover exactly the staged images")
    rows = []
    for image_id in sorted(images):
        image = images[image_id]
        detections = grounding[image_id]["detections"]
        for index, detection in enumerate(detections):
            for family, box in (("grounded_entity", detection["box"]),
                                ("grounded_entity_padded",
                                 padded(detection["box"], image["width"], image["height"]))):
                rows.append({
                    "candidate_id": opaque("cand", image_id, family, box),
                    "staging_image_id": image_id, "kind": "bbox", "boxes": [box],
                    "view_family": family, "grounding_indices": [index],
                    "grounding_labels": [detection["text_label"]],
                    "proposal_score": detection["score"],
                    "proposal_generator_revision": REVISION,
                    "pixel_cost": (box[2] - box[0]) * (box[3] - box[1]),
                })
        pairs = []
        for left in range(len(detections)):
            for right in range(left + 1, len(detections)):
                box = union(detections[left]["box"], detections[right]["box"])
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area >= 0.85 * image["width"] * image["height"]:
                    continue
                pairs.append((center_distance(detections[left]["box"], detections[right]["box"],
                                              image["width"], image["height"]),
                              -min(detections[left]["score"], detections[right]["score"]),
                              left, right, box))
        for _, _, left, right, box in sorted(pairs)[:MAX_RELATION_UNIONS]:
            rows.append({
                "candidate_id": opaque("cand", image_id, "relation_union", box, left, right),
                "staging_image_id": image_id, "kind": "bbox", "boxes": [box],
                "view_family": "relation_union", "grounding_indices": [left, right],
                "grounding_labels": [detections[left]["text_label"],
                                     detections[right]["text_label"]],
                "proposal_score": min(detections[left]["score"], detections[right]["score"]),
                "proposal_generator_revision": REVISION,
                "pixel_cost": (box[2] - box[0]) * (box[3] - box[1]),
            })
        full = [0, 0, image["width"], image["height"]]
        rows.append({
            "candidate_id": opaque("cand", image_id, "full_image", full),
            "staging_image_id": image_id, "kind": "bbox", "boxes": [full],
            "view_family": "full_image", "grounding_indices": [], "grounding_labels": [],
            "proposal_score": None, "proposal_generator_revision": REVISION,
            "pixel_cost": image["width"] * image["height"],
        })
        rows.append({
            "candidate_id": opaque("cand", image_id, "stop"),
            "staging_image_id": image_id, "kind": "stop", "boxes": [],
            "view_family": "stop", "grounding_indices": [], "grounding_labels": [],
            "proposal_score": None, "proposal_generator_revision": REVISION, "pixel_cost": 0,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output, rows)
    counts = {}
    for row in rows:
        counts[row["view_family"]] = counts.get(row["view_family"], 0) + 1
    report = {
        "status": "complete", "images": len(images), "candidates": len(rows),
        "family_counts": counts, "padding_fraction": PADDING_FRACTION,
        "max_relation_unions": MAX_RELATION_UNIONS, "revision": REVISION,
        "grounding_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
