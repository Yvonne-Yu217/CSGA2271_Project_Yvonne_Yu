"""Export a compact method-blind human review packet for the provisional screen."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from acquisition.export_review import reviewer_names, write_csv
from acquisition.observe import crop_candidate, read_jsonl


def blind_id(context_id, candidate_id):
    return "screen_" + hashlib.sha256(f"{context_id}\x1f{candidate_id}".encode()).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--natural-e1-output", type=Path, required=True)
    parser.add_argument("--core-target-output", type=Path, required=True)
    parser.add_argument("--strict-core-target-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-image", type=int, default=4)
    parser.add_argument("--reviewers", default="reviewer_a,reviewer_b")
    parser.add_argument("--seed", type=int, default=2271)
    args = parser.parse_args()
    reviewers = reviewer_names(args.reviewers, parser)
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    natural_report = json.loads((args.natural_e1_output / "provisional_report.json").read_text())
    default_types = {(row["context_id"], row["candidate_id"]): row["label"]
                     for row in read_jsonl(args.core_target_output / "complement_types.jsonl")}
    strict_types = {(row["context_id"], row["candidate_id"]): row["label"]
                    for row in read_jsonl(args.strict_core_target_output / "complement_types.jsonl")}
    if set(default_types) != set(strict_types):
        raise RuntimeError("core-target caches do not cover identical items")
    choices = defaultdict(dict)
    for row in natural_report["context_rows"]:
        if row.get("candidate_id") is not None:
            choices[row["context_id"]][row["method"]] = row["candidate_id"]
    default_report = json.loads((args.core_target_output / "provisional_report.json").read_text())
    for row in default_report["context_rows"]:
        if row.get("candidate_id") is not None:
            choices[row["context_id"]][row["method"]] = row["candidate_id"]
    by_image = defaultdict(dict)
    methods = ("core_target_oracle", "montage_planner", "clip_inverse_cost_matched",
               "largest_area", "coordinate_planner")
    for context_id, method_choices in choices.items():
        context = contexts[context_id]
        for rank, method in enumerate(methods, 1):
            candidate_id = method_choices.get(method)
            if candidate_id is None or candidates[candidate_id]["kind"] == "stop":
                continue
            key = (context_id, candidate_id)
            disagreement = default_types[key] != strict_types[key]
            priority = (0 if disagreement else rank,
                        hashlib.sha256(f"{args.seed}:{context_id}:{candidate_id}".encode()).hexdigest())
            current = by_image[context["staging_image_id"]].get(key)
            if current is None or priority < current:
                by_image[context["staging_image_id"]][key] = priority
    selected = []
    for image_id in sorted(images):
        ranked = sorted(by_image[image_id], key=lambda key: by_image[image_id][key])
        selected.extend(ranked[:args.per_image])
    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    for context_id, candidate_id in selected:
        context, candidate = contexts[context_id], candidates[candidate_id]
        item_id = blind_id(context_id, candidate_id)
        crop_path = assets / f"{item_id}.jpg"
        crop = crop_candidate(args.staging, images[context["staging_image_id"]], candidate)
        crop.save(crop_path, quality=92)
        crop.close()
        rows.append({
            "item_id": item_id, "staging_image_id": context["staging_image_id"],
            "image_path": images[context["staging_image_id"]]["image_path"],
            "crop_path": str(crop_path.relative_to(args.output)),
            "initial_caption": context["initial_caption"],
            "boxes": json.dumps(candidate["boxes"]),
            "candidate_observation": observations[candidate_id]["observed_text"],
            "crop_support": "", "adds_correct_fact": "", "same_instance_as_mentioned_entity": "",
            "complement_type": "", "atomic_correct_facts_json": "", "notes": "",
        })
    manifest = {
        "scope": "method-blind targeted direction screen; blank fields are not annotations",
        "seed": args.seed, "per_image": args.per_image, "images": len(by_image),
        "items": len(rows), "reviewers": reviewers,
        "selection": "method choices/disagreements selected before blinding; methods omitted from review CSV",
        "requirements": ["Review independently.", "Judge the crop before semantic complement type.",
                         "Use unknown when instance linkage is ambiguous."],
    }
    fields = list(rows[0]) if rows else []
    for reviewer_index, reviewer in enumerate(reviewers):
        shuffled = list(rows)
        random.Random(args.seed + reviewer_index).shuffle(shuffled)
        write_csv(args.output / reviewer / "targeted_screen_review.csv", fields, shuffled)
    (args.output / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
