"""Export a blinded R2 review packet spanning population and diagnostic strata."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from acquisition.observe import crop_candidate, read_jsonl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opaque(*parts):
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]


def index_rows(path):
    return {(row["context_id"], row["candidate_id"]): row for row in read_jsonl(path)}


def one_per_image(pool, contexts, seed, limit=None):
    grouped = defaultdict(list)
    for key in pool:
        grouped[contexts[key[0]]["staging_image_id"]].append(key)
    chosen = []
    for image_id in sorted(grouped):
        ranked = sorted(grouped[image_id], key=lambda key: hashlib.sha256(
            f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest())
        chosen.append(ranked[0])
    random.Random(seed).shuffle(chosen)
    return chosen[:limit] if limit else chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--grid-observer", type=Path, required=True)
    parser.add_argument("--grid-qwen", type=Path, required=True)
    parser.add_argument("--grid-smol", type=Path, required=True)
    parser.add_argument("--paraphrase-contexts", type=Path, required=True)
    parser.add_argument("--paraphrase-observer", type=Path, required=True)
    parser.add_argument("--paraphrase-qwen", type=Path, required=True)
    parser.add_argument("--grounded-candidates", type=Path, required=True)
    parser.add_argument("--grounded-observer", type=Path, required=True)
    parser.add_argument("--grounded-qwen", type=Path, required=True)
    parser.add_argument("--grounded-smol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-per-stratum", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2271)
    args = parser.parse_args()

    context_path = args.staging / "natural_contexts.jsonl"
    image_path = args.staging / "staging_images.jsonl"
    grid_candidate_path = args.staging / "automatic_candidates.jsonl"
    contexts = {row["context_id"]: row for row in read_jsonl(context_path)}
    paraphrases = {row["context_id"]: row for row in read_jsonl(args.paraphrase_contexts)}
    images = {row["staging_image_id"]: row for row in read_jsonl(image_path)}
    candidates = {row["candidate_id"]: row for row in read_jsonl(grid_candidate_path)}
    candidates.update({row["candidate_id"]: row for row in read_jsonl(args.grounded_candidates)})
    grid_obs = index_rows(args.grid_observer / "observations.jsonl")
    para_obs = index_rows(args.paraphrase_observer / "observations.jsonl")
    grounded_obs = index_rows(args.grounded_observer / "observations.jsonl")
    grid_qwen = index_rows(args.grid_qwen / "complement_types.jsonl")
    grid_smol = index_rows(args.grid_smol / "complement_types.jsonl")
    para_qwen = index_rows(args.paraphrase_qwen / "complement_types.jsonl")
    grounded_qwen = index_rows(args.grounded_qwen / "complement_types.jsonl")
    grounded_smol = index_rows(args.grounded_smol / "complement_types.jsonl")
    if not (set(grid_obs) == set(grid_qwen) == set(grid_smol)):
        raise RuntimeError("grid caches do not cover identical context-actions")
    if not (set(para_obs) == set(para_qwen) == set(grid_obs)):
        raise RuntimeError("paraphrase caches do not match original grid keys")
    if not (set(grounded_obs) == set(grounded_qwen) == set(grounded_smol)):
        raise RuntimeError("grounded caches do not cover identical context-actions")
    if set(paraphrases) != set(contexts):
        raise RuntimeError("paraphrase contexts do not match natural contexts")

    positive = "MENTIONED_ENTITY_DETAIL"
    population = one_per_image(list(grid_obs), contexts, args.seed)
    disagreement_pool = [key for key in grid_obs
                         if (grid_qwen[key]["label"] == positive) !=
                         (grid_smol[key]["label"] == positive)]
    agreement_pool = [key for key in grid_obs
                      if grid_qwen[key]["label"] == grid_smol[key]["label"] == positive]
    grounded_positive_pool = [key for key in grounded_obs
                              if grounded_qwen[key]["label"] ==
                              grounded_smol[key]["label"] == positive]
    paraphrase_flip_pool = [key for key in grid_obs
                            if (grid_qwen[key]["label"] == positive) !=
                            (para_qwen[key]["label"] == positive)]
    limit = args.diagnostic_per_stratum
    strata = {
        "population_random": population,
        "grid_judge_disagreement": one_per_image(disagreement_pool, contexts, args.seed + 1, limit),
        "grid_consensus_positive": one_per_image(agreement_pool, contexts, args.seed + 2, limit),
        "grounded_consensus_positive": one_per_image(
            grounded_positive_pool, contexts, args.seed + 3, limit),
        "paraphrase_action_flip": one_per_image(
            paraphrase_flip_pool, contexts, args.seed + 4, limit),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    public, private = [], []
    for stratum, keys in strata.items():
        for context_id, candidate_id in keys:
            context = contexts[context_id]
            image_id = context["staging_image_id"]
            candidate = candidates[candidate_id]
            item_id = "r2_" + opaque(str(args.seed), stratum, context_id, candidate_id)
            full_name, crop_name = f"{item_id}_full.jpg", f"{item_id}_crop.jpg"
            shutil.copy2(args.staging / images[image_id]["image_path"], assets / full_name)
            crop = crop_candidate(args.staging, images[image_id], candidate)
            crop.save(assets / crop_name, format="JPEG", quality=95)
            crop.close()
            is_grounded = stratum == "grounded_consensus_positive"
            observation = grounded_obs[(context_id, candidate_id)] if is_grounded else grid_obs[
                (context_id, candidate_id)]
            is_pair = stratum == "paraphrase_action_flip"
            public.append({
                "item_id": item_id,
                "full_image_asset": f"../assets/{full_name}",
                "crop_asset": f"../assets/{crop_name}",
                "existing_caption": context["initial_caption"],
                "candidate_claim": observation["observed_text"],
                "alternate_caption": paraphrases[context_id]["initial_caption"] if is_pair else "",
                "alternate_claim": para_obs[(context_id, candidate_id)]["observed_text"] if is_pair else "",
                "captions_semantically_equivalent_yes_no_uncertain": "",
                "claim_visual_support_yes_no_uncertain": "",
                "claim_adds_new_fact_yes_no_uncertain": "",
                "claim_same_mentioned_instance_detail_yes_no_uncertain": "",
                "alternate_claim_visual_support_yes_no_uncertain": "",
                "alternate_claim_adds_new_fact_yes_no_uncertain": "",
                "alternate_claim_same_mentioned_instance_detail_yes_no_uncertain": "",
                "notes": "",
            })
            private.append({
                "item_id": item_id, "stratum": stratum, "context_id": context_id,
                "staging_image_id": image_id, "candidate_id": candidate_id,
                "qwen_label": (grounded_qwen if is_grounded else grid_qwen)[
                    (context_id, candidate_id)]["label"],
                "smol_label": (grounded_smol if is_grounded else grid_smol)[
                    (context_id, candidate_id)]["label"],
                "paraphrase_qwen_label": para_qwen[(context_id, candidate_id)]["label"]
                if is_pair else None,
            })
    fields = list(public[0])
    reviewer_orders = {}
    for reviewer_index, reviewer in enumerate(("reviewer_a", "reviewer_b"), 1):
        rows = list(public)
        random.Random(args.seed + reviewer_index).shuffle(rows)
        reviewer_orders[reviewer] = [row["item_id"] for row in rows]
        reviewer_dir = args.output / reviewer
        reviewer_dir.mkdir(parents=True, exist_ok=True)
        with (reviewer_dir / "r2_blind_review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (args.output / "private_manifest.jsonl").open("w") as handle:
        for row in sorted(private, key=lambda item: item["item_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_paths = [context_path, image_path, grid_candidate_path, args.paraphrase_contexts,
                    args.grounded_candidates, args.grid_observer / "observations.jsonl",
                    args.paraphrase_observer / "observations.jsonl",
                    args.grounded_observer / "observations.jsonl",
                    args.grid_qwen / "complement_types.jsonl",
                    args.grid_smol / "complement_types.jsonl",
                    args.paraphrase_qwen / "complement_types.jsonl",
                    args.grounded_qwen / "complement_types.jsonl",
                    args.grounded_smol / "complement_types.jsonl"]
    label_fields = [field for field in fields if field.endswith("yes_no_uncertain")]
    audit = {
        "unique_public_item_ids": len({row["item_id"] for row in public}) == len(public),
        "public_private_item_ids_match": ({row["item_id"] for row in public} ==
                                          {row["item_id"] for row in private}),
        "reviewer_orders_differ": reviewer_orders["reviewer_a"] != reviewer_orders["reviewer_b"],
        "all_review_fields_blank": all(not row[field] for row in public for field in label_fields),
        "all_assets_exist": all((args.output / "reviewer_a" / row[field]).resolve().is_file()
                                for row in public
                                for field in ("full_image_asset", "crop_asset")),
        "automated_labels_private_only": not any(
            field in fields for field in ("stratum", "qwen_label", "smol_label",
                                          "paraphrase_qwen_label")),
    }
    if not all(audit.values()):
        raise RuntimeError(f"review packet audit failed: {audit}")
    summary = {
        "status": "ready_for_two_independent_human_reviewers",
        "scope": "blank method-blind R2 development review; not annotations",
        "seed": args.seed, "review_rows_per_reviewer": len(public),
        "stratum_counts": {key: len(value) for key, value in strata.items()},
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "audit": audit,
        "requirements": [
            "Review independently before adjudication.",
            "Use uncertain rather than guessing instance identity or visual support.",
            "Do not inspect private_manifest.jsonl or automated results while reviewing.",
            "Population-random rows estimate error; enriched strata diagnose failures and are not prevalence samples.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    instructions = """# R2 blinded review instructions

Open your own `r2_blind_review.csv`; do not open the other reviewer's CSV or
`private_manifest.jsonl` until both independent reviews are complete.

For every non-empty judgment field enter exactly `yes`, `no`, or `uncertain`.
Judge visual support from the supplied full image and crop. Judge novelty
against the associated existing caption. Mark same-instance detail `yes` only
when the claim concerns the same specific entity instance named in that
caption. For paired paraphrase rows, also judge whether the two captions state
the same facts; then judge both candidate claims separately. Leave the
alternate-claim fields blank when no alternate claim is supplied.

The population-random rows estimate error rates. The remaining rows are
diagnostic strata and must not be used as prevalence estimates. Blank fields
are unfinished reviews, never negative labels.
"""
    (args.output / "REVIEW_INSTRUCTIONS.md").write_text(instructions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
