"""Evaluate frozen prompted planners against R2 automated strict action labels."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--strict-output", type=Path, required=True)
    parser.add_argument("--coordinate-planner", type=Path, required=True)
    parser.add_argument("--montage-planner", type=Path, required=True)
    parser.add_argument("--full-image-core-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    labels = read_jsonl(args.strict_output / "complement_types.jsonl")
    success = {(row["context_id"], row["candidate_id"]):
               int(row["label"] == "MENTIONED_ENTITY_DETAIL") for row in labels}
    choices = {
        "coordinate_planner": {row["context_id"]: row for row in read_jsonl(
            args.coordinate_planner / "choices.jsonl")},
        "montage_planner": {row["context_id"]: row for row in read_jsonl(
            args.montage_planner / "choices.jsonl")},
    }
    if any(set(rows) != set(contexts) for rows in choices.values()):
        raise RuntimeError("planner choices must cover exactly all contexts")
    candidates_by_image, contexts_by_image = defaultdict(set), defaultdict(set)
    for context_id, context in contexts.items():
        contexts_by_image[context["staging_image_id"]].add(context_id)
    for context_id, candidate_id in success:
        candidates_by_image[contexts[context_id]["staging_image_id"]].add(candidate_id)
    ordered_context_ids, oracle, static, clusters = [], [], [], []
    for image_id in sorted(contexts_by_image):
        context_ids = sorted(contexts_by_image[image_id])
        candidate_ids = sorted(candidates_by_image[image_id])
        best_static = max(candidate_ids, key=lambda candidate_id: (
            sum(success[(context_id, candidate_id)] for context_id in context_ids), candidate_id))
        for context_id in context_ids:
            ordered_context_ids.append(context_id)
            oracle.append(max(success[(context_id, candidate_id)] for candidate_id in candidate_ids))
            static.append(success[(context_id, best_static)])
            clusters.append(image_id)
    full_image = {row["context_id"]: row for row in read_jsonl(
        args.full_image_core_output / "complement_types.jsonl")}
    vectors = {
        "recognition_oracle": oracle,
        "best_same_image_static": static,
        "full_image_completion": [
            int(full_image[context_id]["label"] == "MENTIONED_ENTITY_DETAIL")
            for context_id in ordered_context_ids],
    }
    for method, rows in choices.items():
        vectors[method] = [
            success.get((context_id, rows[context_id]["candidate_id"]), 0)
            for context_id in ordered_context_ids]
    methods = {
        method: bootstrap_cluster(values, clusters, samples=10000, seed=2271)
        for method, values in vectors.items()
    }
    gaps = {
        f"{method}_minus_full_image": paired_bootstrap(
            values, vectors["full_image_completion"], clusters, samples=10000, seed=2271)
        for method, values in vectors.items() if method != "full_image_completion"
    }
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated strict action labels are not human adjudication.",
        "contexts": len(ordered_context_ids), "images": len(set(clusters)),
        "methods": methods, "paired_gaps": gaps,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.strict_output / "complement_types.jsonl",
                         args.coordinate_planner / "choices.jsonl",
                         args.montage_planner / "choices.jsonl",
                         args.full_image_core_output / "complement_types.jsonl")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
