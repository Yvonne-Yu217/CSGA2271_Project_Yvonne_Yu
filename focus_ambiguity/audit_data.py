"""Audit the official VQ-FocusAmbiguity images, metadata, and segmentation masks."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors, warnings = [], []
    warning_examples = defaultdict(list)
    records_by_split = {}
    for split in SPLITS:
        path = args.source / f"{split}_gt.json"
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            errors.append(f"{path} is not a JSON list")
            records = []
        records_by_split[split] = records
    ids, mask_ids = {}, {}
    image_splits = defaultdict(set)
    image_records = defaultdict(list)
    label_counts = Counter()
    source_counts = Counter()
    internal_set_counts = Counter()
    label_mask_counts = Counter()
    mask_count = 0
    image_paths = {}
    json_size_mismatch = 0
    json_size_swapped = 0
    image_mask_size_mismatch = 0
    internal_set_mismatch = 0
    for split, records in records_by_split.items():
        for row in records:
            row_id, masks_id = row.get("id"), row.get("masks_id")
            internal_set_counts[(split, row.get("set"))] += 1
            if row.get("set") != split:
                internal_set_mismatch += 1
                if len(warning_examples["internal_set_mismatch"]) < 20:
                    warning_examples["internal_set_mismatch"].append({
                        "id": row_id, "file_split": split, "internal_set": row.get("set")
                    })
            if row_id in ids:
                errors.append(f"duplicate global id {row_id}: {ids[row_id]} and {split}")
            ids[row_id] = split
            if masks_id in mask_ids:
                errors.append(f"duplicate masks_id {masks_id}: {mask_ids[masks_id]} and {split}")
            mask_ids[masks_id] = split
            if row.get("label") not in {"ambiguous", "unambiguous"}:
                errors.append(f"unknown label for id {row_id}: {row.get('label')}")
            label_counts[(split, row.get("label"))] += 1
            source_counts[(split, row.get("attributes", {}).get("data_source"))] += 1
            file_name = row.get("file_name")
            image_splits[file_name].add(split)
            image_records[file_name].append({
                "split": split,
                "id": row_id,
                "question": row.get("question", ""),
            })
            image_path = args.extracted / "images" / file_name
            image_paths[file_name] = image_path
            if not image_path.is_file():
                errors.append(f"missing image for id {row_id}: {file_name}")
                continue
            with Image.open(image_path) as image:
                image_size = image.size
                expected_size = (row["size"]["width"], row["size"]["height"])
                if image_size != expected_size:
                    json_size_mismatch += 1
                    kind = "json_size_swapped" if image_size == expected_size[::-1] else "json_size_other"
                    json_size_swapped += kind == "json_size_swapped"
                    if len(warning_examples[kind]) < 20:
                        warning_examples[kind].append({
                            "id": row_id, "split": split, "file_name": file_name,
                            "image_size": image_size, "json_size": expected_size,
                        })
            mask_dir = args.extracted / "masks_gt_v3" / split / masks_id
            masks = sorted(mask_dir.glob("*.png")) if mask_dir.is_dir() else []
            expected_masks = len(row.get("focus_areas", []))
            if len(masks) != expected_masks:
                errors.append(f"mask count mismatch id {row_id}: {len(masks)} != {expected_masks}")
            # The archive deliberately renames masks to 0000.png, 0001.png, ...;
            # focus_areas retains the masks' original descriptive filenames.
            focus_names = row.get("focus_areas", [])
            if len(set(focus_names)) != len(focus_names):
                errors.append(f"duplicate focus-area filename id {row_id}")
            if any(not name.endswith(".png") for name in focus_names):
                errors.append(f"non-PNG focus-area filename id {row_id}")
            label_mask_counts[(row.get("label"), expected_masks)] += 1
            mask_count += len(masks)
            for mask_path in masks:
                with Image.open(mask_path) as mask:
                    if mask.size != image_size:
                        image_mask_size_mismatch += 1
                        errors.append(
                            f"image/mask size mismatch id {row_id}: {image_size} != {mask.size}"
                        )
                    extrema = mask.convert("L").getextrema()
                    if extrema is None or extrema[1] == 0:
                        errors.append(f"empty mask id {row_id}: {mask_path.name}")
                    colors = mask.convert("L").getcolors(maxcolors=257)
                    values = {value for _, value in colors} if colors else set()
                    if not values.issubset({0, 255}):
                        errors.append(f"non-binary mask id {row_id}: {sorted(values)[:10]}")
    cross_split_images = {name: sorted(splits) for name, splits in image_splits.items()
                          if len(splits) > 1}
    if cross_split_images:
        warnings.append(
            f"{len(cross_split_images)} image filenames cross official splits; "
            "group by image for any internal model-selection split"
        )
    cross_split_records = {
        name: image_records[name] for name in sorted(cross_split_images)
    }
    exact_cross_split_question_duplicates = []
    for file_name, rows in cross_split_records.items():
        grouped = defaultdict(list)
        for row in rows:
            normalized = " ".join(row["question"].lower().split())
            grouped[normalized].append(row)
        for normalized, duplicate_rows in grouped.items():
            if len({row["split"] for row in duplicate_rows}) > 1:
                exact_cross_split_question_duplicates.append({
                    "file_name": file_name,
                    "normalized_question": normalized,
                    "records": duplicate_rows,
                })
    if exact_cross_split_question_duplicates:
        warnings.append(
            f"{len(exact_cross_split_question_duplicates)} exact image-question pairs cross official splits"
        )
    if internal_set_mismatch:
        warnings.append(
            f"{internal_set_mismatch} records have an internal set field inconsistent with their JSON file"
        )
    archive_image_names = {path.name for path in (args.extracted / "images").iterdir()
                           if path.is_file() and not path.name.startswith("._")}
    referenced_image_names = set(image_paths)
    missing_references = sorted(referenced_image_names - archive_image_names)
    unused_images = sorted(archive_image_names - referenced_image_names)
    if missing_references:
        errors.append(f"{len(missing_references)} referenced image files are absent")
    if unused_images:
        warnings.append(f"{len(unused_images)} extracted images are not referenced")
    source_files = [args.source / f"{split}_gt.json" for split in SPLITS] + [
        args.source / "focusAmbiguity_gts.zip", args.source / "fa_images.zip"]
    report = {
        "status": "fail" if errors else ("pass_with_warnings" if warnings or json_size_mismatch else "pass"),
        "records": {split: len(records_by_split[split]) for split in SPLITS},
        "records_total": sum(map(len, records_by_split.values())),
        "unique_images": len(referenced_image_names),
        "masks": mask_count,
        "label_counts": {f"{split}:{label}": count
                         for (split, label), count in sorted(label_counts.items())},
        "source_counts": {f"{split}:{source}": count
                          for (split, source), count in sorted(source_counts.items())},
        "label_mask_count_joint": {f"{label}:{count}": total
                                   for (label, count), total in sorted(label_mask_counts.items())},
        "dimension_checks": {
            "json_size_mismatch_records": json_size_mismatch,
            "json_size_swapped_records": json_size_swapped,
            "image_mask_size_mismatch_masks": image_mask_size_mismatch,
        },
        "internal_set_mismatch_records": internal_set_mismatch,
        "internal_set_counts": {f"file_{split}:field_{field}": count
                                for (split, field), count in sorted(internal_set_counts.items())},
        "cross_split_images": cross_split_images,
        "cross_split_image_count": len(cross_split_images),
        "cross_split_records": cross_split_records,
        "exact_cross_split_image_question_duplicates": exact_cross_split_question_duplicates,
        "warning_examples": dict(warning_examples),
        "unused_images": unused_images,
        "source_sha256": {path.name: sha256(path) for path in source_files},
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"cross_split_images", "cross_split_records",
                                     "exact_cross_split_image_question_duplicates",
                                     "label_mask_count_joint", "unused_images",
                                     "warning_examples"}}, indent=2,
                     sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
