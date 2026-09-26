"""Generate a deterministic two-image E0--E2 fixture for tests and smoke runs."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from acquisition import SCHEMA_VERSION
from acquisition.schema import GoldStore, PublicStore, derive_action_labels, observation_cache_key


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_fixture(root):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    image_rows = []
    for index, image_id in enumerate(("im1", "im2"), 1):
        path = root / "images" / f"{image_id}.png"
        Image.new("RGB", (10, 10), (20 * index, 30 * index, 40 * index)).save(path)
        image_rows.append({
            "image_id": image_id, "image_path": f"images/{image_id}.png",
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 10, "height": 10, "source": "synthetic-fixture",
            "source_image_id": image_id, "duplicate_group_id": f"dup_{image_id}", "split": "dev",
        })

    contexts = [
        {"context_id": "c1_sparse", "image_id": "im1", "initial_caption": "A person rides a bicycle.",
         "source_kind": "authored", "context_kind": "sparse", "provenance_id": "fixture",
         "context_group_id": "g1"},
        {"context_id": "c1_rear", "image_id": "im1", "initial_caption": "A person rides with a child behind them.",
         "source_kind": "authored", "context_kind": "enriched", "provenance_id": "fixture",
         "context_group_id": "g1", "reference_context_id": "c1_sparse"},
        {"context_id": "c1_para", "image_id": "im1", "initial_caption": "A cyclist carries a child on the back seat.",
         "source_kind": "authored", "context_kind": "paraphrase", "provenance_id": "fixture",
         "context_group_id": "g1", "reference_context_id": "c1_rear"},
        {"context_id": "c1_full", "image_id": "im1", "initial_caption": "A red-clad cyclist carries a child behind them and flowers in a basket.",
         "source_kind": "authored", "context_kind": "saturated", "provenance_id": "fixture",
         "context_group_id": "g1", "reference_context_id": "c1_rear"},
        {"context_id": "c2_sparse", "image_id": "im2", "initial_caption": "Two people stand outside.",
         "source_kind": "authored", "context_kind": "sparse", "provenance_id": "fixture",
         "context_group_id": "g2"},
        {"context_id": "c2_full", "image_id": "im2", "initial_caption": "The person on the left hands a bag to the person on the right.",
         "source_kind": "authored", "context_kind": "saturated", "provenance_id": "fixture",
         "context_group_id": "g2", "reference_context_id": "c2_sparse"},
    ]

    candidates = []
    def candidate(candidate_id, image_id, kind, boxes, score):
        candidates.append({
            "candidate_id": candidate_id, "image_id": image_id, "kind": kind, "boxes": boxes,
            "proposal_generator_revision": "fixture-grid-v1",
            "pixel_cost": sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes),
            "proposal_score": score,
        })
    candidate("r1_clothes", "im1", "bbox", [[0, 0, 5, 5]], 0.9)
    candidate("r1_basket", "im1", "bbox", [[5, 0, 10, 5]], 0.7)
    candidate("r1_rear", "im1", "bbox", [[0, 5, 10, 10]], 0.6)
    candidate("r1_stop", "im1", "stop", [], 0.0)
    candidate("r2_left", "im2", "bbox", [[0, 0, 5, 10]], 0.9)
    candidate("r2_right", "im2", "bbox", [[5, 0, 10, 10]], 0.8)
    candidate("r2_joint", "im2", "region_set", [[0, 0, 5, 10], [5, 0, 10, 10]], 0.5)
    candidate("r2_stop", "im2", "stop", [], 0.0)

    observed = {
        "r1_clothes": "The rider wears red.", "r1_basket": "Flowers are in the basket.",
        "r1_rear": "A child sits behind the rider.", "r1_stop": "",
        "r2_left": "A person stands on the left.", "r2_right": "A person stands on the right.",
        "r2_joint": "The left person hands a bag to the right person.", "r2_stop": "",
    }
    observations = []
    by_image = {row["image_id"]: row for row in image_rows}
    for row in candidates:
        prompt_hash = hashlib.sha256(b"fixture observer prompt v1").hexdigest()
        observations.append({
            "observation_id": f"o_{row['candidate_id']}", "image_id": row["image_id"],
            "candidate_id": row["candidate_id"], "observer_revision": "fixture-observer-v1",
            "prompt_hash": prompt_hash,
            "cache_key": observation_cache_key(by_image[row["image_id"]]["image_sha256"], row,
                                                "fixture-observer-v1", prompt_hash),
            "observed_text": observed[row["candidate_id"]],
            "output_tokens": len(observed[row["candidate_id"]].split()), "latency_ms": 1,
            "status": "stop" if row["kind"] == "stop" else "ok",
        })

    entities = [
        {"entity_id": "e1_rider", "image_id": "im1", "category": "person",
         "evidence_regions": [[0, 0, 5, 10]], "annotation_status": "confirmed"},
        {"entity_id": "e1_child", "image_id": "im1", "category": "person",
         "evidence_regions": [[5, 5, 10, 10]], "annotation_status": "confirmed"},
        {"entity_id": "e1_basket", "image_id": "im1", "category": "basket",
         "evidence_regions": [[5, 0, 10, 5]], "annotation_status": "confirmed"},
        {"entity_id": "e2_left", "image_id": "im2", "category": "person",
         "evidence_regions": [[0, 0, 5, 10]], "annotation_status": "confirmed"},
        {"entity_id": "e2_right", "image_id": "im2", "category": "person",
         "evidence_regions": [[5, 0, 10, 10]], "annotation_status": "confirmed"},
    ]
    facts = [
        {"fact_id": "f1_rider", "image_id": "im1", "fact_type": "action",
         "subject_entity_id": "e1_rider", "predicate": "riding", "object": {"literal": "bicycle"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f1_red", "image_id": "im1", "fact_type": "attribute",
         "subject_entity_id": "e1_rider", "predicate": "wearing_color", "object": {"literal": "red"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f1_flower", "image_id": "im1", "fact_type": "attribute",
         "subject_entity_id": "e1_basket", "predicate": "contains", "object": {"literal": "flowers"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f1_child", "image_id": "im1", "fact_type": "entity",
         "subject_entity_id": "e1_child", "predicate": "present", "object": {"literal": "yes"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f1_rear", "image_id": "im1", "fact_type": "relation",
         "subject_entity_id": "e1_child", "predicate": "behind", "object": {"entity_id": "e1_rider"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f2_left", "image_id": "im2", "fact_type": "entity",
         "subject_entity_id": "e2_left", "predicate": "present", "object": {"literal": "yes"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f2_right", "image_id": "im2", "fact_type": "entity",
         "subject_entity_id": "e2_right", "predicate": "present", "object": {"literal": "yes"},
         "polarity": "positive", "visual_support": "yes"},
        {"fact_id": "f2_hands", "image_id": "im2", "fact_type": "relation",
         "subject_entity_id": "e2_left", "predicate": "hands_bag_to", "object": {"entity_id": "e2_right"},
         "polarity": "positive", "visual_support": "yes"},
    ]
    fact_ids = {row["image_id"]: [fact["fact_id"] for fact in facts if fact["image_id"] == row["image_id"]]
                for row in image_rows}
    entailed = {
        "c1_sparse": {"f1_rider"},
        "c1_rear": {"f1_rider", "f1_child", "f1_rear"},
        "c1_para": {"f1_rider", "f1_child", "f1_rear"},
        "c1_full": set(fact_ids["im1"]),
        "c2_sparse": {"f2_left", "f2_right"},
        "c2_full": set(fact_ids["im2"]),
    }
    context_fact = []
    for context in contexts:
        for fact_id in fact_ids[context["image_id"]]:
            status = "entailed" if fact_id in entailed[context["context_id"]] else "not_entailed"
            context_fact.append({
                "context_id": context["context_id"], "fact_id": fact_id, "status": status,
                "mention_links": [], "annotator_ids": ["reviewer_a", "reviewer_b"],
                "annotator_votes": [status, status], "adjudication": status,
            })

    visible = {
        "r1_clothes": {"f1_red"}, "r1_basket": {"f1_flower"},
        "r1_rear": {"f1_child", "f1_rear"}, "r1_stop": set(),
        "r2_left": {"f2_left"}, "r2_right": {"f2_right"},
        "r2_joint": {"f2_left", "f2_right", "f2_hands"}, "r2_stop": set(),
    }
    candidate_fact = []
    for row in candidates:
        for fact_id in fact_ids[row["image_id"]]:
            yes = fact_id in visible[row["candidate_id"]]
            candidate_fact.append({
                "candidate_id": row["candidate_id"], "fact_id": fact_id,
                "visibility": "full" if yes else "none", "sufficient_to_verify": "yes" if yes else "no",
                "annotator_ids": ["reviewer_a", "reviewer_b"],
                "visibility_annotator_votes": ["full" if yes else "none"] * 2,
                "visibility_adjudication": "full" if yes else "none",
                "annotator_votes": ["yes" if yes else "no"] * 2,
                "adjudication": "yes" if yes else "no",
            })

    claim_rows = []
    for candidate_id, candidate_facts in visible.items():
        for fact_id in sorted(candidate_facts):
            claim_rows.append({
                "claim_id": f"cl_{candidate_id}_{fact_id}", "observation_id": f"o_{candidate_id}",
                "mapped_fact_id": fact_id, "claim_text": observed[candidate_id], "status": "correct",
                "annotator_ids": ["reviewer_a", "reviewer_b"],
                "annotator_votes": ["correct", "correct"], "adjudication": "correct",
            })

    context_metadata = [{
        "context_id": row["context_id"], "context_kind": row.pop("context_kind"),
        "context_group_id": row.pop("context_group_id"),
        "reference_context_id": row.pop("reference_context_id", None),
    } for row in contexts]

    # Public IDs are intentionally opaque even in the fixture, so tests cannot
    # accidentally teach selectors semantic information through identifiers.
    def opaque(namespace, value):
        return namespace + "_" + hashlib.sha256(value.encode()).hexdigest()[:20]
    image_map = {row["image_id"]: opaque("img", row["image_id"]) for row in image_rows}
    context_map = {row["context_id"]: opaque("ctx", row["context_id"]) for row in contexts}
    candidate_map = {row["candidate_id"]: opaque("cand", row["candidate_id"])
                     for row in candidates}
    observation_map = {row["observation_id"]: opaque("obs", row["observation_id"])
                       for row in observations}
    for row in image_rows:
        old = row["image_id"]
        row["image_id"] = image_map[old]
        row["source_image_id"] = opaque("src", row["source_image_id"])
        row["duplicate_group_id"] = opaque("dup", row["duplicate_group_id"])
    for row in contexts:
        row["context_id"] = context_map[row["context_id"]]
        row["image_id"] = image_map[row["image_id"]]
    for row in context_metadata:
        row["context_id"] = context_map[row["context_id"]]
        row["context_group_id"] = opaque("grp", row["context_group_id"])
        if row["reference_context_id"] is not None:
            row["reference_context_id"] = context_map[row["reference_context_id"]]
    for row in candidates:
        row["candidate_id"] = candidate_map[row["candidate_id"]]
        row["image_id"] = image_map[row["image_id"]]
    for row in observations:
        row["observation_id"] = observation_map[row["observation_id"]]
        row["candidate_id"] = candidate_map[row["candidate_id"]]
        row["image_id"] = image_map[row["image_id"]]
    for row in entities + facts:
        row["image_id"] = image_map[row["image_id"]]
    for row in context_fact:
        row["context_id"] = context_map[row["context_id"]]
    for row in candidate_fact:
        row["candidate_id"] = candidate_map[row["candidate_id"]]
    for row in claim_rows:
        row["observation_id"] = observation_map[row["observation_id"]]

    for name, rows in (("images", image_rows), ("contexts", contexts), ("candidates", candidates),
                       ("observations", observations)):
        _write_jsonl(root / "public" / f"{name}.jsonl", rows)
    for name, rows in (("context_metadata", context_metadata), ("entities", entities), ("facts", facts),
                       ("context_fact_labels", context_fact),
                       ("candidate_fact_labels", candidate_fact),
                       ("observation_claim_labels", claim_rows), ("action_labels", [])):
        _write_jsonl(root / "gold" / f"{name}.jsonl", rows)
    (root / "provenance.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "automatic_candidate_generator_revision": "fixture-grid-v1",
        "unresolved_cross_split_near_duplicates": [],
        "note": "Synthetic test fixture only; never research evidence.",
    }, indent=2) + "\n")
    public, gold = PublicStore(root), GoldStore(root)
    actions = derive_action_labels(public, gold)
    _write_jsonl(root / "gold" / "action_labels.jsonl",
                 [actions[key] for key in sorted(actions)])
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.output)
    print(args.output)
