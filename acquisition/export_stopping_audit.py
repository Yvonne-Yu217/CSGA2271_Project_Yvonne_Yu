"""Export a method-blind human stopping audit for direct completion versus crop oracle."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from acquisition.observe import crop_candidate, read_jsonl


DEFAULT_STRATA = {"direct_only": 30, "oracle_only": 30, "both": 10, "neither": 10}


def opaque_id(*parts):
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--full-image-screen-output", type=Path, required=True)
    parser.add_argument("--full-image-core-output", type=Path, required=True)
    parser.add_argument("--candidate-core-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2271)
    args = parser.parse_args()
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    completions = {row["context_id"]: row for row in read_jsonl(
        args.full_image_screen_output / "judgments.jsonl")}
    direct_types = {row["context_id"]: row for row in read_jsonl(
        args.full_image_core_output / "complement_types.jsonl")}
    candidate_report = json.loads(
        (args.candidate_core_output / "provisional_report.json").read_text())
    oracle_rows = {row["context_id"]: row for row in candidate_report["context_rows"]
                   if row["method"] == "core_target_oracle"}
    expected = set(contexts)
    if any(set(table) != expected for table in (completions, direct_types, oracle_rows)):
        raise RuntimeError("stopping-audit sources do not cover the same contexts")
    strata = defaultdict(list)
    for context_id in sorted(contexts):
        direct = direct_types[context_id]["label"] == "MENTIONED_ENTITY_DETAIL"
        oracle = bool(oracle_rows[context_id]["core_target_success"])
        name = ("both" if direct and oracle else "direct_only" if direct else
                "oracle_only" if oracle else "neither")
        strata[name].append(context_id)
    rng = random.Random(args.seed)
    selected = []
    for name, count in DEFAULT_STRATA.items():
        pool = list(strata[name])
        rng.shuffle(pool)
        if len(pool) < count:
            raise RuntimeError(f"stratum {name} has {len(pool)} rows, need {count}")
        selected.extend((context_id, name) for context_id in pool[:count])
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    private_rows, public_rows = [], []
    for context_id, stratum in selected:
        context = contexts[context_id]
        image_row = images[context["staging_image_id"]]
        oracle_candidate_id = oracle_rows[context_id]["candidate_id"]
        oracle_candidate = candidates[oracle_candidate_id]
        claims = [
            ("direct_full_image", completions[context_id]["completion"], None),
            ("candidate_oracle", observations[oracle_candidate_id]["observed_text"],
             oracle_candidate),
        ]
        for method, claim, candidate in claims:
            review_id = "audit_" + opaque_id(str(args.seed), context_id, method)
            asset_name = review_id + ".jpg"
            asset_path = assets / asset_name
            if candidate is None:
                shutil.copy2(args.staging / image_row["image_path"], asset_path)
            else:
                crop = crop_candidate(args.staging, image_row, candidate)
                crop.save(asset_path, format="JPEG", quality=95)
                crop.close()
            public_rows.append({
                "review_id": review_id,
                "image_asset": f"assets/{asset_name}",
                "existing_caption": context["initial_caption"],
                "candidate_claim": claim,
                "visual_support_yes_no_uncertain": "",
                "adds_new_fact_yes_no_uncertain": "",
                "same_mentioned_instance_detail_yes_no_uncertain": "",
                "claim_atomic_yes_no_uncertain": "",
                "notes": "",
            })
            private_rows.append({
                "review_id": review_id, "context_id": context_id,
                "staging_image_id": context["staging_image_id"], "method": method,
                "candidate_id": oracle_candidate_id if candidate else None,
                "automated_stratum": stratum,
            })
    fieldnames = list(public_rows[0])
    for reviewer, reviewer_seed in (("reviewer_1", args.seed + 1),
                                     ("reviewer_2", args.seed + 2)):
        rows = list(public_rows)
        random.Random(reviewer_seed).shuffle(rows)
        with (args.output / f"{reviewer}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with (args.output / "private_manifest.jsonl").open("w") as handle:
        for row in sorted(private_rows, key=lambda item: item["review_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "status": "ready_for_two_independent_human_reviewers",
        "contexts": len(selected), "review_rows_per_reviewer": len(public_rows),
        "strata_requested": DEFAULT_STRATA,
        "available_strata": {key: len(value) for key, value in sorted(strata.items())},
        "seed": args.seed,
        "warning": "Reviewer CSVs contain no method identity or automated label.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
