"""Evaluate E1 oracle space and E2 caption dependence on an audited bundle."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from acquisition.metrics import (
    action_metrics, bootstrap_cluster, caption_necessity, choose_best,
    expected_random_metrics, material_switch, pair_regret, paired_bootstrap,
)
from acquisition.schema import GoldStore, PublicStore, derive_action_labels, validate_bundle
from acquisition.audit import audit_report


def _candidate_center_score(candidate, image):
    if not candidate["boxes"]:
        return float("-inf")
    cx, cy = image["width"] / 2, image["height"] / 2
    distances = []
    for x1, y1, x2, y2 in candidate["boxes"]:
        distances.append((0.5 * (x1 + x2) - cx) ** 2 + (0.5 * (y1 + y2) - cy) ** 2)
    return -min(distances)


def evaluate_bundle(root, baseline_method="proposal_score", bootstrap_samples=10000, seed=2271,
                    strong_baseline_manifest=None):
    audit = validate_bundle(root)
    if audit["status"] != "pass":
        raise ValueError(f"bundle audit failed: {audit['errors'][:3]}")
    public, gold = PublicStore(root), GoldStore(root)
    metadata = {row["context_id"]: row for row in gold.tables["context_metadata"]}
    actions = derive_action_labels(public, gold)
    fact_count = defaultdict(int)
    for fact in gold.tables["facts"]:
        if fact["visual_support"] == "yes":
            fact_count[fact["image_id"]] += 1
    candidates_by_image = defaultdict(list)
    for candidate in public.candidates.values():
        candidates_by_image[candidate["image_id"]].append(candidate)

    context_rows = []
    score_vectors = defaultdict(dict)
    for context in public.contexts.values():
        image_id = context["image_id"]
        image = public.images[image_id]
        candidates = candidates_by_image[image_id]
        ids = [candidate["candidate_id"] for candidate in candidates]
        lookup = {candidate_id: actions[(context["context_id"], candidate_id)]
                  for candidate_id in ids}
        selected = {
            "recognition_oracle": choose_best(ids, lookup, fact_count[image_id]),
            "largest_area": max(candidates, key=lambda row: (row["pixel_cost"], row["candidate_id"]))["candidate_id"],
            "proposal_score": max(candidates, key=lambda row: (row.get("proposal_score", float("-inf")),
                                                                 row["candidate_id"]))["candidate_id"],
            "center": max(candidates, key=lambda row: (_candidate_center_score(row, image),
                                                        row["candidate_id"]))["candidate_id"],
            "stop": next(row["candidate_id"] for row in candidates if row["kind"] == "stop"),
        }
        for candidate_id in ids:
            score_vectors[context["context_id"]][candidate_id] = action_metrics(
                lookup[candidate_id], fact_count[image_id])["delta_coverage"]
        random_metrics = expected_random_metrics(ids, lookup, fact_count[image_id])
        for method, candidate_id in selected.items():
            metrics = action_metrics(lookup[candidate_id], fact_count[image_id])
            context_rows.append({
                "image_id": image_id, "duplicate_group_id": image["duplicate_group_id"],
                "context_id": context["context_id"],
                "context_kind": metadata[context["context_id"]]["context_kind"],
                "method": method, "candidate_id": candidate_id, **metrics,
            })
        context_rows.append({
            "image_id": image_id, "duplicate_group_id": image["duplicate_group_id"],
            "context_id": context["context_id"],
            "context_kind": metadata[context["context_id"]]["context_kind"],
            "method": "random_expected", "candidate_id": None, **random_metrics,
        })

    methods = sorted({row["method"] for row in context_rows})
    metrics = ("delta_coverage", "new_correct_count", "uncovered_fact_recall",
               "invalid_claim_count", "invalid_claim_rate", "repeat_count", "repeat_rate",
               "unknown_claim_count", "pixel_cost")
    summary = {}
    for method in methods:
        selected = [row for row in context_rows if row["method"] == method]
        # Contexts are first macro-averaged within image, then duplicate groups
        # are resampled. Images with more caption variants cannot dominate.
        image_rows = defaultdict(list)
        for row in selected:
            image_rows[row["image_id"]].append(row)
        summary[method] = {
            metric: bootstrap_cluster(
                                      [mean for image_id in sorted(image_rows)
                                       if (mean := _finite_mean(r[metric] for r in image_rows[image_id]))
                                       is not None],
                                      [public.images[image_id]["duplicate_group_id"]
                                       for image_id in sorted(image_rows)
                                       if _finite_mean(r[metric] for r in image_rows[image_id]) is not None],
                                      samples=bootstrap_samples, seed=seed)
            for metric in metrics
        }

    if baseline_method not in methods or baseline_method in {"recognition_oracle", "random_expected"}:
        raise ValueError(f"baseline_method must be a deployable deterministic method: {baseline_method}")
    by_key = {(row["context_id"], row["method"]): row for row in context_rows}
    contexts = sorted(public.contexts)
    oracle_values = [by_key[(context, "recognition_oracle")]["delta_coverage"] for context in contexts]
    baseline_values = [by_key[(context, baseline_method)]["delta_coverage"] for context in contexts]
    clusters = [public.images[public.contexts[context]["image_id"]]["duplicate_group_id"]
                for context in contexts]
    oracle_gap = paired_bootstrap(oracle_values, baseline_values, clusters,
                                  samples=bootstrap_samples, seed=seed)
    oracle_errors = [by_key[(context, "recognition_oracle")]["invalid_claim_count"] for context in contexts]
    baseline_errors = [by_key[(context, baseline_method)]["invalid_claim_count"] for context in contexts]
    error_gap = paired_bootstrap(oracle_errors, baseline_errors, clusters,
                                 samples=bootstrap_samples, seed=seed)

    by_image_contexts = defaultdict(list)
    for context in public.contexts.values():
        by_image_contexts[context["image_id"]].append(context)
    necessity_rows, switch_rows, paraphrase_rows, saturated_rows = [], [], [], []
    context_known = defaultdict(set)
    for row in gold.tables["context_fact_labels"]:
        if row["status"] == "entailed":
            context_known[row["context_id"]].add(row["fact_id"])
    for image_id, image_contexts in by_image_contexts.items():
        groups = defaultdict(list)
        for context in image_contexts:
            groups[metadata[context["context_id"]]["context_group_id"]].append(context)
        # Primary necessity uses every eligible context exactly once per image.
        gaps = []
        if len(image_contexts) >= 2:
            all_scores = {row["context_id"]: score_vectors[row["context_id"]]
                          for row in image_contexts}
            gaps.append(caption_necessity(all_scores)["caption_necessity_gap"])
        image_switches = []
        for group_contexts in groups.values():
            if len(group_contexts) < 2:
                continue
            for index, left in enumerate(group_contexts):
                for right in group_contexts[index + 1:]:
                    left_kind = metadata[left["context_id"]]["context_kind"]
                    right_kind = metadata[right["context_id"]]["context_kind"]
                    if (left_kind != "paraphrase" and right_kind != "paraphrase" and
                            context_known[left["context_id"]] != context_known[right["context_id"]]):
                        image_switches.append(float(material_switch(
                            score_vectors[left["context_id"]], score_vectors[right["context_id"]])))
        if gaps:
            necessity_rows.append((sum(gaps) / len(gaps), public.images[image_id]["duplicate_group_id"]))
        if image_switches:
            switch_rows.append((sum(image_switches) / len(image_switches),
                                public.images[image_id]["duplicate_group_id"]))
        for context in image_contexts:
            context_meta = metadata[context["context_id"]]
            if context_meta["context_kind"] == "paraphrase" and context_meta["reference_context_id"]:
                reference = context_meta["reference_context_id"]
                paraphrase_rows.append((pair_regret(score_vectors[reference],
                                                     score_vectors[context["context_id"]]),
                                        public.images[image_id]["duplicate_group_id"]))
            action = by_key[(context["context_id"], baseline_method)]
            eligible = actions[(context["context_id"], action["candidate_id"])]["eligible_missing_fact_ids"]
            if not eligible:
                saturated_rows.append((float(public.candidates[action["candidate_id"]]["kind"] == "stop"),
                                       public.images[image_id]["duplicate_group_id"]))

    necessity = bootstrap_cluster([row[0] for row in necessity_rows], [row[1] for row in necessity_rows],
                                  samples=bootstrap_samples, seed=seed)
    paraphrase = bootstrap_cluster([row[0] for row in paraphrase_rows], [row[1] for row in paraphrase_rows],
                                   samples=bootstrap_samples, seed=seed)
    stop_accuracy = bootstrap_cluster([row[0] for row in saturated_rows],
                                      [row[1] for row in saturated_rows],
                                      samples=bootstrap_samples, seed=seed)
    material_switch_rate = bootstrap_cluster(
        [row[0] for row in switch_rows], [row[1] for row in switch_rows],
        samples=bootstrap_samples, seed=seed)
    e0 = audit_report(root)["e0_gate"]["eligible"]
    strong_baseline = False
    if strong_baseline_manifest:
        manifest = json.loads(Path(strong_baseline_manifest).read_text())
        # This evaluator currently contains only diagnostic built-ins. A manifest
        # cannot relabel one of them as a strong baseline; external score-row
        # ingestion and identity checks must be implemented first.
        diagnostic = {"proposal_score", "largest_area", "center", "stop"}
        strong_baseline = bool(baseline_method not in diagnostic and
                               manifest.get("frozen_before_evaluation") and
                               manifest.get("complete_cost_accounting") and
                               manifest.get("method") == baseline_method)
    evidence_eligible = e0 and strong_baseline
    model_paraphrase_stability_available = False
    return {
        "protocol": {
            "primary_unit": "duplicate_group_id", "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed, "frozen_baseline_method": baseline_method,
            "oracle": "recognition oracle over actual frozen-observer outputs; evaluation-only",
            "e0_eligible": e0, "strong_baseline_verified": strong_baseline,
            "evidence_eligible": evidence_eligible,
        },
        "e1": {
            "methods": summary,
            "oracle_minus_frozen_baseline": oracle_gap,
            "invalid_rate_oracle_minus_baseline": error_gap,
            "decision_thresholds": {
                "minimum_oracle_gap": 0.10, "oracle_gap_ci_lower_must_exceed": 0.0,
                "invalid_rate_ci_upper_maximum": 0.02,
            },
            "gate_pass": evidence_eligible and oracle_gap["mean"] is not None and oracle_gap["mean"] >= 0.10 and
                         oracle_gap["ci95"][0] > 0 and
                         error_gap["ci95"][1] is not None and error_gap["ci95"][1] <= 0.02,
        },
        "e2": {
            "caption_necessity_gap": necessity,
            "material_switch_rate": material_switch_rate,
            "paraphrase_gold_action_regret_data_check": paraphrase,
            "model_paraphrase_stability_available": model_paraphrase_stability_available,
            "saturated_stop_accuracy": stop_accuracy,
            "decision_thresholds": {
                "minimum_necessity_gap": 0.03, "necessity_ci_lower_must_exceed": 0.0,
                "minimum_material_switch_rate": 0.20,
                "paraphrase_regret_ci_upper_maximum": 0.02,
                "minimum_stop_accuracy": 0.80,
            },
            "gate_pass": evidence_eligible and necessity["mean"] is not None and necessity["mean"] >= 0.03 and
                         necessity["ci95"][0] > 0 and
                         material_switch_rate["mean"] is not None and
                         material_switch_rate["mean"] >= 0.20 and
                         model_paraphrase_stability_available and
                         paraphrase["ci95"][1] is not None and paraphrase["ci95"][1] < 0.02 and
                         stop_accuracy["mean"] is not None and stop_accuracy["mean"] >= 0.80,
        },
        "context_rows": context_rows,
}


def _finite_mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-method", default="proposal_score",
                        choices=("proposal_score", "largest_area", "center", "stop"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2271)
    parser.add_argument("--strong-baseline-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_bundle(args.bundle, args.baseline_method, args.bootstrap_samples, args.seed,
                             args.strong_baseline_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
