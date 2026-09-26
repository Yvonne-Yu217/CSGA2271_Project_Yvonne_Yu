"""Validate an E0 acquisition bundle without exposing gold to selectors."""
import argparse
import json
from pathlib import Path

from acquisition.schema import GoldStore, PublicStore, validate_bundle


def agreement_summary(root):
    public = PublicStore(root)
    gold = GoldStore(root)
    context_image = {row["context_id"]: row["image_id"] for row in public.contexts.values()}
    candidate_image = {row["candidate_id"]: row["image_id"] for row in public.candidates.values()}
    observation_image = {row["observation_id"]: row["image_id"] for row in public.observations.values()}
    fact_type = {row["fact_id"]: row["fact_type"] for row in gold.tables["facts"]}
    rows = []
    for table in ("context_fact_labels", "candidate_fact_labels", "observation_claim_labels"):
        for row in gold.tables[table]:
            if table == "context_fact_labels":
                image_id = context_image[row["context_id"]]
                vote_sets = [("context_status", row.get("annotator_votes", []))]
            elif table == "candidate_fact_labels":
                image_id = candidate_image[row["candidate_id"]]
                vote_sets = [
                    ("candidate_visibility", row.get("visibility_annotator_votes", [])),
                    ("candidate_sufficiency", row.get("annotator_votes", [])),
                ]
            else:
                image_id = observation_image[row["observation_id"]]
                vote_sets = [("observation_claim_status", row.get("annotator_votes", []))]
            for family, votes in vote_sets:
                if len(votes) < 2:
                    continue
                linked_fact = row.get("fact_id", row.get("mapped_fact_id"))
                adjudicated = (row.get("visibility_adjudication") is not None
                               if family == "candidate_visibility"
                               else row.get("adjudication") is not None)
                rows.append((family, image_id, votes[0], votes[1], adjudicated,
                             fact_type.get(linked_fact, "unmapped")))
    def stats(selected):
        labels = sorted({row[2] for row in selected} | {row[3] for row in selected})
        agreement = sum(row[2] == row[3] for row in selected) / len(selected) if selected else None
        if not selected:
            kappa = None
        else:
            left = {label: sum(row[2] == label for row in selected) / len(selected) for label in labels}
            right = {label: sum(row[3] == label for row in selected) / len(selected) for label in labels}
            chance = sum(left[label] * right[label] for label in labels)
            kappa = (agreement - chance) / (1 - chance) if chance < 1 else None
        return {
            "double_annotated_rows": len(selected),
            "double_annotated_images": len({row[1] for row in selected}),
            "raw_agreement": agreement, "cohen_kappa": kappa,
            "adjudication_rate": sum(row[4] for row in selected) / len(selected) if selected else None,
        }
    by_table = {}
    for table in sorted({row[0] for row in rows}):
        selected = [row for row in rows if row[0] == table]
        by_table[table] = stats(selected)
    by_fact_type = {}
    for kind in sorted({row[5] for row in rows}):
        by_fact_type[kind] = stats([row for row in rows if row[5] == kind])
    return {"overall": stats(rows), "by_table": by_table, "by_fact_type": by_fact_type}


def audit_report(root):
    report = validate_bundle(root)
    if report["status"] != "pass":
        report["e0_gate"] = {"eligible": False, "reason": "schema audit failed"}
        return report
    report["annotation_agreement"] = agreement_summary(root)
    public = PublicStore(root)
    required = ("context_status", "candidate_visibility", "candidate_sufficiency",
                "observation_claim_status")
    per_table = report["annotation_agreement"]["by_table"]
    sources = {row["source"] for row in public.images.values()}
    synthetic = any("synthetic" in source.lower() or "fixture" in source.lower()
                    for source in sources)
    table_checks = {
        table: {
            "at_least_100_images": per_table.get(table, {}).get("double_annotated_images", 0) >= 100,
            "raw_agreement_at_least_0_90": (
                per_table.get(table, {}).get("raw_agreement") is not None and
                per_table[table]["raw_agreement"] >= 0.90),
            "all_rows_adjudicated": per_table.get(table, {}).get("adjudication_rate") == 1.0,
        } for table in required
    }
    eligible = (not synthetic and all(all(checks.values()) for checks in table_checks.values()))
    report["e0_gate"] = {
        "eligible": eligible,
        "synthetic_or_fixture": synthetic,
        "minimum_double_annotated_images_per_table": 100,
        "minimum_raw_agreement_per_table": 0.90,
        "table_checks": table_checks,
        "note": "Every label family gates separately; adjudication never replaces raw agreement.",
    }
    report["status"] = "eligible" if eligible else "not_eligible"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_report(args.bundle)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    raise SystemExit(0 if report["status"] == "eligible" else 2)


if __name__ == "__main__":
    main()
