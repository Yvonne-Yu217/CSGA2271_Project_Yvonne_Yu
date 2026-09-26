"""Mechanical audit for review-only E2 context proposals."""
import argparse
import json
import re
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--proposal-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    rows = read_jsonl(args.proposal_output / "context_proposals.jsonl")
    metadata = json.loads((args.proposal_output / "runtime.json").read_text())
    errors, flags = [], []
    if metadata.get("status") != "complete":
        errors.append("proposal metadata is not complete")
    if len(rows) != len({row.get("staging_image_id") for row in rows}):
        errors.append("duplicate proposal image rows")
    if {row.get("staging_image_id") for row in rows} != set(images):
        errors.append("proposal rows do not cover every staged image exactly once")
    for row in rows:
        image_id = row.get("staging_image_id")
        reference_id = row.get("reference_context_id")
        proposed = row.get("proposals")
        if reference_id not in contexts or contexts[reference_id]["staging_image_id"] != image_id:
            errors.append(f"invalid reference context: {row.get('proposal_id')}")
            continue
        if not isinstance(proposed, dict) or set(proposed) != {"enriched", "paraphrase", "saturated"}:
            errors.append(f"missing proposal fields: {row.get('proposal_id')}")
            continue
        reference = row["reference_caption"]
        if reference != contexts[reference_id]["initial_caption"]:
            errors.append(f"reference caption mismatch: {row.get('proposal_id')}")
        for kind, text in proposed.items():
            words = re.findall(r"\b\w+\b", text)
            if not text.strip() or len(words) > 80:
                flags.append({"proposal_id": row["proposal_id"], "kind": kind,
                              "flag": "empty_or_over_80_words"})
            if text.strip().casefold() == reference.strip().casefold():
                flags.append({"proposal_id": row["proposal_id"], "kind": kind,
                              "flag": "identical_to_reference"})
        enriched, paraphrase = tokens(proposed["enriched"]), tokens(proposed["paraphrase"])
        union = enriched | paraphrase
        similarity = len(enriched & paraphrase) / len(union) if union else 1.0
        if similarity > 0.90 or similarity < 0.20:
            flags.append({"proposal_id": row["proposal_id"], "kind": "paraphrase",
                          "flag": "lexical_similarity_outside_0.20_0.90",
                          "jaccard": similarity})
    report = {
        "status": "fail" if errors else "pass_with_review_flags" if flags else "pass",
        "scope": "mechanical proposal audit only; visual correctness requires independent review",
        "counts": {"images": len(images), "proposal_rows": len(rows), "review_flags": len(flags)},
        "errors": errors, "review_flags": flags,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "counts", "errors")}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
