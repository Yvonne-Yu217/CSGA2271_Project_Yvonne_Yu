"""Export two independent, method-blind E0 review stages."""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from acquisition.schema import PublicStore


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def blind_id(*parts):
    return "item_" + hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]


def reviewer_names(raw, parser):
    names = [value.strip() for value in raw.split(",") if value.strip()]
    if len(names) < 2 or len(names) != len(set(names)):
        parser.error("at least two unique reviewer names are required")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in names):
        parser.error("reviewer names may contain only letters, digits, underscore, and hyphen")
    return names


def fact_stage(staging):
    images = {row["staging_image_id"]: row
              for row in read_jsonl(staging / "staging_images.jsonl")}
    evidence = read_jsonl(staging / "review_input" / "source_entity_evidence.jsonl")
    rows = []
    for row in evidence:
        rows.append({
            "item_id": blind_id("fact", row["review_item_id"]),
            "staging_image_id": row["staging_image_id"],
            "image_path": images[row["staging_image_id"]]["image_path"],
            "caption": row["caption"], "source_phrase": row["phrase"],
            "source_types": json.dumps(row["source_types"]),
            "source_boxes": json.dumps(row["boxes"]),
            "instance_key": "", "normalized_fact": "", "fact_type": "",
            "subject_instance": "", "predicate": "", "object_instance_or_literal": "",
            "visual_support": "", "annotation_status": "", "notes": "",
        })
    return {"fact_inventory": rows}, {"images": len(images), "fact_seed_rows": len(rows)}


def label_stage(bundle):
    public = PublicStore(bundle)
    fact_path = Path(bundle) / "gold" / "facts.jsonl"
    facts = read_jsonl(fact_path)
    fact_ids = [row.get("fact_id") for row in facts]
    if len(fact_ids) != len(set(fact_ids)) or any(not value for value in fact_ids):
        raise RuntimeError("frozen fact inventory has missing or duplicate fact IDs")
    if any(row.get("image_id") not in public.images for row in facts):
        raise RuntimeError("frozen fact inventory references an unknown image")
    facts_by_image = {}
    for fact in facts:
        facts_by_image.setdefault(fact["image_id"], []).append(fact["fact_id"])
    contexts = []
    for row in public.contexts.values():
        contexts.append({
            "item_id": blind_id("context", row["context_id"]),
            "context_id": row["context_id"], "image_id": row["image_id"],
            "image_path": public.images[row["image_id"]]["image_path"],
            "initial_caption": row["initial_caption"],
            "canonical_fact_ids_json": json.dumps(sorted(facts_by_image[row["image_id"]])),
            "caption_correctness": "", "resolved_instance_mentions_json": "",
            "fact_statuses_json": "", "notes": "",
        })
    claims = []
    for observation in public.observations.values():
        if observation["status"] != "ok":
            continue
        candidate = public.candidates[observation["candidate_id"]]
        claims.append({
            "item_id": blind_id("claim", observation["observation_id"]),
            "observation_id": observation["observation_id"],
            "observer_revision": observation["observer_revision"],
            "observer_prompt_hash": observation["prompt_hash"],
            "image_id": observation["image_id"],
            "image_path": public.images[observation["image_id"]]["image_path"],
            "candidate_id": candidate["candidate_id"], "boxes": json.dumps(candidate["boxes"]),
            "observed_text": observation["observed_text"],
            "canonical_fact_ids_json": json.dumps(sorted(facts_by_image[observation["image_id"]])),
            "atomic_claims_json": "", "claim_statuses_json": "",
            "mapped_fact_ids_json": "", "unmatched_claims_json": "", "notes": "",
        })
    candidate_facts = []
    for candidate in public.candidates.values():
        for fact_id in sorted(facts_by_image[candidate["image_id"]]):
            candidate_facts.append({
                "item_id": blind_id("candidate-fact", candidate["candidate_id"], fact_id),
                "image_id": candidate["image_id"],
                "image_path": public.images[candidate["image_id"]]["image_path"],
                "candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
                "boxes": json.dumps(candidate["boxes"]), "canonical_fact_id": fact_id,
                "visibility": "", "sufficient_to_verify": "", "notes": "",
            })
    return {"context_coverage": contexts, "candidate_facts": candidate_facts,
            "observation_claims": claims}, {
        "images": len(public.images), "context_rows": len(contexts),
        "candidate_fact_rows": len(candidate_facts), "claim_rows": len(claims),
        "canonical_fact_inventory_sha256": hashlib.sha256(fact_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("facts", "labels"))
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--canonical-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewers", default="reviewer_a,reviewer_b")
    parser.add_argument("--seed", type=int, default=6203)
    args = parser.parse_args()
    reviewers = reviewer_names(args.reviewers, parser)
    if args.stage == "facts":
        if args.staging is None:
            parser.error("--staging is required for facts stage")
        tables, counts = fact_stage(args.staging)
    else:
        if args.canonical_bundle is None:
            parser.error("--canonical-bundle is required for labels stage")
        tables, counts = label_stage(args.canonical_bundle)
    manifest = {
        "scope": "method-blind blank review templates; not annotations",
        "stage": args.stage, "seed": args.seed, "reviewers": reviewers, **counts,
        "requirements": [
            "Reviewers work independently before adjudication.",
            "Unknown is explicit and blank fields are never interpreted as labels.",
            "No method scores or another reviewer's file may be inspected.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for reviewer_index, reviewer in enumerate(reviewers):
        for name, rows in tables.items():
            shuffled = list(rows)
            random.Random(rng.randrange(2 ** 32) + reviewer_index).shuffle(shuffled)
            write_csv(args.output / reviewer / f"{name}.csv",
                      list(shuffled[0]) if shuffled else [], shuffled)
    (args.output / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
