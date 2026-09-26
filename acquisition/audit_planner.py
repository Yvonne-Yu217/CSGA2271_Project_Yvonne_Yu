"""Audit completeness, validity, and cost distribution of a planner cache."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--planner-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    choices = read_jsonl(args.planner_output / "choices.jsonl")
    metadata = json.loads((args.planner_output / "runtime.json").read_text())
    errors = []
    by_context = {row.get("context_id"): row for row in choices}
    if len(by_context) != len(choices):
        errors.append("duplicate context choices")
    if set(by_context) != set(contexts):
        errors.append("choices do not cover every natural context exactly once")
    selected = []
    for row in choices:
        candidate = candidates.get(row.get("candidate_id"))
        context = contexts.get(row.get("context_id"))
        if row.get("status") != "ok" or candidate is None or context is None:
            errors.append(f"invalid choice row: {row.get('context_id')}")
            continue
        if candidate["staging_image_id"] != context["staging_image_id"]:
            errors.append(f"cross-image choice: {row.get('context_id')}")
            continue
        selected.append(candidate)
    if metadata.get("status") != "complete" or metadata.get("completed") != len(contexts):
        errors.append("planner metadata is not complete")
    kind_counts = Counter(row["kind"] for row in selected)
    costs = [row["pixel_cost"] for row in selected]
    report = {
        "status": "fail" if errors else "pass",
        "scope": "cache audit only; choice utility requires adjudicated E0 outcomes",
        "fingerprint": metadata.get("fingerprint"),
        "visualization": metadata.get("visualization", "coordinates"),
        "counts": {"contexts": len(contexts), "choice_rows": len(choices),
                   "valid_choices": len(selected), "selected_kinds": dict(kind_counts)},
        "cost": {"pixel_cost_total": sum(costs),
                 "pixel_cost_mean": sum(costs) / len(costs) if costs else None,
                 "pixel_cost_max": max(costs) if costs else None},
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
