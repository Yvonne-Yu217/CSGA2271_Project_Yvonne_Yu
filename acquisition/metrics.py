"""E1/E2 metrics with duplicate-group paired bootstrap."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def mean_or_none(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def action_metrics(action, fact_count):
    new_count = len(action["new_supported_fact_ids"])
    invalid_count = len(action["contradicted_claim_ids"])
    repeat_count = len(action["repeated_fact_ids"])
    eligible = len(action["eligible_missing_fact_ids"])
    return {
        "delta_coverage": new_count / fact_count if fact_count else None,
        "new_correct_count": new_count,
        "uncovered_fact_recall": new_count / eligible if eligible else None,
        "invalid_claim_count": invalid_count,
        "invalid_claim_rate": invalid_count / (new_count + invalid_count)
        if new_count + invalid_count else None,
        "repeat_count": repeat_count,
        "repeat_rate": repeat_count / (new_count + repeat_count)
        if new_count + repeat_count else None,
        "unknown_claim_count": len(action["unknown_claim_ids"]),
        "pixel_cost": action["pixel_cost"],
    }


def choose_best(candidate_ids, action_lookup, fact_count):
    """Recognition oracle: maximize measured coverage, then fewer errors/cost, then ID."""
    def key(candidate_id):
        metrics = action_metrics(action_lookup[candidate_id], fact_count)
        return (
            metrics["delta_coverage"] if metrics["delta_coverage"] is not None else -1,
            -metrics["invalid_claim_count"], -metrics["pixel_cost"], candidate_id,
        )
    return max(candidate_ids, key=key)


def expected_random_metrics(candidate_ids, action_lookup, fact_count):
    rows = [action_metrics(action_lookup[candidate_id], fact_count) for candidate_id in candidate_ids]
    result = {key: mean_or_none(row[key] for row in rows) for key in rows[0]}
    expected_new = result["new_correct_count"]
    expected_invalid = result["invalid_claim_count"]
    expected_repeat = result["repeat_count"]
    invalid_denominator = expected_new + expected_invalid
    repeat_denominator = expected_new + expected_repeat
    result["invalid_claim_rate"] = (expected_invalid / invalid_denominator
                                    if invalid_denominator else None)
    result["repeat_rate"] = expected_repeat / repeat_denominator if repeat_denominator else None
    return result


def bootstrap_cluster(values, clusters, *, samples=10000, seed=2271):
    """Percentile interval after resampling whole duplicate groups."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters differ in length")
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        if value is not None and math.isfinite(value):
            grouped[cluster].append(float(value))
    group_ids = sorted(grouped)
    if not group_ids:
        return {"mean": None, "ci95": [None, None], "clusters": 0, "samples": samples,
                "seed": seed}
    group_values = np.asarray([np.mean(grouped[group]) for group in group_ids], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(group_values), size=(samples, len(group_values)))
    estimates = group_values[draws].mean(axis=1)
    return {
        "mean": float(group_values.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "clusters": len(group_values), "samples": samples, "seed": seed,
    }


def paired_bootstrap(left, right, clusters, *, samples=10000, seed=2271):
    if not (len(left) == len(right) == len(clusters)):
        raise ValueError("paired inputs differ in length")
    differences, kept_clusters = [], []
    for a, b, cluster in zip(left, right, clusters):
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            continue
        differences.append(float(a) - float(b))
        kept_clusters.append(cluster)
    return bootstrap_cluster(differences, kept_clusters, samples=samples, seed=seed)


def epsilon_top_set(scores, epsilon=0.01):
    if not scores:
        return set()
    maximum = max(scores.values())
    return {key for key, value in scores.items() if maximum - value <= epsilon}


def caption_necessity(image_context_scores):
    """Compare a caption-specific oracle with the best fixed action per image.

    Input is ``{context_id: {candidate_id: delta_coverage}}`` and every context
    must share exactly the same candidate IDs.
    """
    contexts = sorted(image_context_scores)
    if not contexts:
        return None
    candidate_sets = [set(image_context_scores[context]) for context in contexts]
    if any(candidates != candidate_sets[0] for candidates in candidate_sets[1:]):
        raise ValueError("paired contexts must have identical candidate sets")
    adaptive = float(np.mean([max(image_context_scores[context].values()) for context in contexts]))
    fixed = max(float(np.mean([image_context_scores[context][candidate]
                               for context in contexts])) for candidate in candidate_sets[0])
    return {"caption_specific_value": adaptive, "best_static_value": fixed,
            "caption_necessity_gap": adaptive - fixed}


def pair_regret(reference_scores, comparison_scores, epsilon=0.01):
    """Regret on comparison context after choosing any reference top action.

    The conservative value uses the worst comparison outcome among epsilon-tied
    reference optima so candidate ordering cannot improve the result.
    """
    if set(reference_scores) != set(comparison_scores):
        raise ValueError("paired contexts must have identical candidate sets")
    chosen = epsilon_top_set(reference_scores, epsilon)
    return max(comparison_scores.values()) - min(comparison_scores[candidate]
                                                  for candidate in chosen)


def material_switch(left_scores, right_scores, *, epsilon=0.01, material_loss=0.05):
    left_top = epsilon_top_set(left_scores, epsilon)
    right_top = epsilon_top_set(right_scores, epsilon)
    if left_top & right_top:
        return False
    left_regret = max(right_scores.values()) - max(right_scores[c] for c in left_top)
    right_regret = max(left_scores.values()) - max(left_scores[c] for c in right_top)
    return max(left_regret, right_regret) >= material_loss
