"""Convert completed/adjudicated R2 human sheets into gated R3 action targets."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from acquisition.audit_r2_reviews import BASE_FIELDS, PAIR_FIELDS, read_csv, required_fields
from acquisition.observe import atomic_jsonl, read_jsonl


def resolved_vote(left, right, adjudication):
    left, right = left.strip().lower(), right.strip().lower()
    if left not in {"yes", "no", "uncertain"} or right not in {"yes", "no", "uncertain"}:
        raise ValueError("both reviewer votes must be complete and valid")
    if left == right:
        return left
    value = (adjudication or "").strip().lower()
    if value not in {"yes", "no", "uncertain"}:
        raise ValueError("disagreeing reviewer votes require a valid adjudication")
    return value


def action_target(component_votes):
    values = list(component_votes)
    if len(values) != 3 or any(value not in {"yes", "no", "uncertain"} for value in values):
        raise ValueError("action target requires three resolved component votes")
    if "no" in values:
        return "negative"
    if all(value == "yes" for value in values):
        return "positive"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.packet / "review_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("run acquisition.audit_r2_reviews first")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "ready_for_adjudication":
        raise RuntimeError(f"human review gate closed: {audit.get('status')}")
    reviewer_a = {row["item_id"]: row for row in read_csv(
        args.packet / "reviewer_a" / "r2_blind_review.csv")}
    reviewer_b = {row["item_id"]: row for row in read_csv(
        args.packet / "reviewer_b" / "r2_blind_review.csv")}
    private_rows = {row["item_id"]: row for row in read_jsonl(
        args.packet / "private_manifest.jsonl")}
    if not (set(reviewer_a) == set(reviewer_b) == set(private_rows)):
        raise RuntimeError("public/private review item sets differ")
    with (args.packet / "adjudication_required.csv").open(newline="") as handle:
        adjudication_rows = list(csv.DictReader(handle))
    adjudications = {(row["item_id"], row["field"]):
                     row["adjudication_yes_no_uncertain"].strip().lower()
                     for row in adjudication_rows}
    targets = []
    for item_id in sorted(private_rows):
        left, right = reviewer_a[item_id], reviewer_b[item_id]
        metadata = private_rows[item_id]
        resolved = {}
        for field in required_fields(left):
            resolved[field] = resolved_vote(
                left[field], right[field], adjudications.get((item_id, field)))
        target = action_target(resolved[field] for field in BASE_FIELDS)
        targets.append({
            "target_id": f"{item_id}:original", "item_id": item_id,
            "context_id": metadata["context_id"], "candidate_id": metadata["candidate_id"],
            "staging_image_id": metadata["staging_image_id"], "stratum": metadata["stratum"],
            "caption_variant": "original", "target": target,
            "component_adjudications": {field: resolved[field] for field in BASE_FIELDS},
            "paraphrase_consistency_eligible": (
                resolved.get(PAIR_FIELDS[0]) == "yes" if PAIR_FIELDS[0] in resolved else False),
        })
        if left.get("alternate_claim", "").strip():
            alternate_components = PAIR_FIELDS[1:]
            targets.append({
                "target_id": f"{item_id}:paraphrase", "item_id": item_id,
                "context_id": metadata["context_id"],
                "candidate_id": metadata["candidate_id"],
                "staging_image_id": metadata["staging_image_id"],
                "stratum": metadata["stratum"], "caption_variant": "paraphrase",
                "target": action_target(resolved[field] for field in alternate_components),
                "component_adjudications": {
                    field: resolved[field] for field in alternate_components},
                "paraphrase_consistency_eligible": resolved[PAIR_FIELDS[0]] == "yes",
            })
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output / "human_action_targets.jsonl", targets)
    summary = {
        "status": "complete_human_targets",
        "scope": "adjudicated human R2 action targets; sampled development rows only",
        "items": len(private_rows), "target_rows": len(targets),
        "target_counts": dict(Counter(row["target"] for row in targets)),
        "variant_counts": dict(Counter(row["caption_variant"] for row in targets)),
        "stratum_counts": dict(Counter(row["stratum"] for row in targets)),
        "consistency_eligible_rows": sum(row["paraphrase_consistency_eligible"]
                                         for row in targets),
        "warning": "Sampling strata must be respected; enriched rows do not estimate prevalence.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
