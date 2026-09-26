"""Audit two completed R2 review sheets and export human-vote disagreements."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABEL_FIELDS = (
    "captions_semantically_equivalent_yes_no_uncertain",
    "claim_visual_support_yes_no_uncertain",
    "claim_adds_new_fact_yes_no_uncertain",
    "claim_same_mentioned_instance_detail_yes_no_uncertain",
    "alternate_claim_visual_support_yes_no_uncertain",
    "alternate_claim_adds_new_fact_yes_no_uncertain",
    "alternate_claim_same_mentioned_instance_detail_yes_no_uncertain",
)
BASE_FIELDS = LABEL_FIELDS[1:4]
PAIR_FIELDS = (LABEL_FIELDS[0],) + LABEL_FIELDS[4:]
ALLOWED = {"", "yes", "no", "uncertain"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def required_fields(row):
    return BASE_FIELDS + PAIR_FIELDS if row.get("alternate_claim", "").strip() else BASE_FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    reviewer_paths = {
        name: args.packet / name / "r2_blind_review.csv"
        for name in ("reviewer_a", "reviewer_b")
    }
    tables = {name: read_csv(path) for name, path in reviewer_paths.items()}
    indexed = {name: {row["item_id"]: row for row in rows}
               for name, rows in tables.items()}
    errors = []
    expected = set(indexed["reviewer_a"])
    if len(expected) != len(tables["reviewer_a"]):
        errors.append("reviewer_a has duplicate item IDs")
    if set(indexed["reviewer_b"]) != expected or len(expected) != len(tables["reviewer_b"]):
        errors.append("reviewer item ID sets differ or reviewer_b has duplicates")
    invalid = []
    completion = {}
    for reviewer, rows in tables.items():
        filled, required = 0, 0
        for row in rows:
            for field in LABEL_FIELDS:
                value = row.get(field, "").strip().lower()
                if value not in ALLOWED:
                    invalid.append({"reviewer": reviewer, "item_id": row["item_id"],
                                    "field": field, "value": value})
            fields = required_fields(row)
            required += len(fields)
            filled += sum(bool(row.get(field, "").strip()) for field in fields)
        completion[reviewer] = {"filled_required_fields": filled,
                                "required_fields": required,
                                "rate": filled / required if required else 0.0}
    if invalid:
        errors.append(f"invalid labels: {len(invalid)}")
    comparisons, disagreements = Counter(), []
    if not errors:
        for item_id in sorted(expected):
            left, right = indexed["reviewer_a"][item_id], indexed["reviewer_b"][item_id]
            for field in required_fields(left):
                a, b = left[field].strip().lower(), right[field].strip().lower()
                if a and b:
                    comparisons[(field, "compared")] += 1
                    comparisons[(field, "agreed" if a == b else "disagreed")] += 1
                    if a != b:
                        disagreements.append({
                            "item_id": item_id, "field": field,
                            "reviewer_a_vote": a, "reviewer_b_vote": b,
                            "adjudication_yes_no_uncertain": "", "adjudicator_notes": "",
                        })
    fully_complete = all(value["rate"] == 1.0 for value in completion.values())
    status = "fail" if errors else "ready_for_adjudication" if fully_complete else "awaiting_reviews"
    report = {
        "status": status, "items": len(expected), "completion": completion,
        "invalid_labels": invalid[:20], "errors": errors,
        "agreement": {
            field: {
                "compared": comparisons[(field, "compared")],
                "agreed": comparisons[(field, "agreed")],
                "disagreed": comparisons[(field, "disagreed")],
                "rate": (comparisons[(field, "agreed")] / comparisons[(field, "compared")]
                         if comparisons[(field, "compared")] else None),
            } for field in LABEL_FIELDS
        },
        "disagreements": len(disagreements),
        "warning": "This audit uses only human sheets; adjudication remains blank and separate.",
    }
    (args.packet / "review_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["item_id", "field", "reviewer_a_vote", "reviewer_b_vote",
              "adjudication_yes_no_uncertain", "adjudicator_notes"]
    with (args.packet / "adjudication_required.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("review audit failed")


if __name__ == "__main__":
    main()
