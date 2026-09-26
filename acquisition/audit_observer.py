"""Audit completeness and formatting of a staged observer cache."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    observations = read_jsonl(args.observer_output / "observations.jsonl")
    metadata = json.loads((args.observer_output / "observer_metadata.json").read_text())
    by_candidate = {row["candidate_id"]: row for row in observations}
    errors = []
    if len(by_candidate) != len(observations):
        errors.append("duplicate candidate observations")
    if set(by_candidate) != set(candidates):
        errors.append("observation/candidate coverage mismatch")
    if metadata.get("status") != "complete":
        errors.append("observer metadata is not complete")
    ok = [row for row in observations if row["status"] == "ok"]
    stops = [row for row in observations if row["status"] == "stop"]
    if any(candidates[row["candidate_id"]]["kind"] != "stop" for row in stops):
        errors.append("non-STOP candidate has stop observation")
    if any(candidates[row["candidate_id"]]["kind"] == "stop" for row in ok):
        errors.append("STOP candidate has generated observation")
    max_tokens = metadata["max_new_tokens"]
    truncated = [row["candidate_id"] for row in ok if row["output_tokens"] >= max_tokens]
    empty = [row["candidate_id"] for row in ok if not row["observed_text"].strip()]
    over_30 = [row["candidate_id"] for row in ok if len(row["observed_text"].split()) > 30]
    duplicates = Counter(row["observed_text"] for row in ok)
    report = {
        "status": "fail" if errors else "pass_with_review_flags" if truncated or empty or over_30 else "pass",
        "scope": "Cache/instruction audit only; model claims remain unverified until blind human review.",
        "observer_fingerprint": metadata["fingerprint"],
        "counts": {
            "candidates": len(candidates), "observations": len(observations), "ok": len(ok),
            "stop": len(stops), "empty": len(empty),
            "no_visible_fact": sum(row["observed_text"] == "NO_VISIBLE_FACT" for row in ok),
            "max_token_truncation": len(truncated), "over_30_words": len(over_30),
            "unique_generated_texts": len(duplicates),
        },
        "review_flags": {
            "empty_candidate_ids": empty, "max_token_candidate_ids": truncated,
            "over_30_word_candidate_ids": over_30,
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "counts", "errors")}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
