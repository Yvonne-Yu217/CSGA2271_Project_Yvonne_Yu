"""Mechanical and leakage audit for the label-free R3-v2 feature store."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from acquisition.observe import read_jsonl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-output", type=Path, required=True)
    parser.add_argument("--grounded-candidates", type=Path, required=True)
    args = parser.parse_args()
    store_path = args.store_output / "feature_store.pt"
    runtime_path = args.store_output / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    store = torch.load(store_path, map_location="cpu", weights_only=True)
    candidates = {row["candidate_id"]: row for row in read_jsonl(args.grounded_candidates)
                  if row.get("view_family") in {"grounded_entity", "full_image", "stop"}}
    errors = []
    actions, feature_dim = store["candidate_features"].shape
    contexts = len(store["context_ids"])
    images = len(store["image_ids"])
    if runtime.get("store_sha256") != digest(store_path):
        errors.append("store hash differs from runtime manifest")
    if store.get("fingerprint") != runtime.get("fingerprint"):
        errors.append("store/runtime fingerprints differ")
    if (actions, feature_dim, contexts, images) != (
            runtime.get("actions"), runtime.get("candidate_feature_dim"),
            runtime.get("contexts"), runtime.get("images")):
        errors.append("runtime shape counts differ from feature store")
    if len(store["candidate_ids"]) != actions or len(set(store["candidate_ids"])) != actions:
        errors.append("candidate IDs are missing or duplicated")
    if set(store["candidate_ids"]) != set(candidates):
        errors.append("feature store candidate IDs differ from frozen action inventory")
    if len(store["context_image_ids"]) != contexts or len(set(store["context_ids"])) != contexts:
        errors.append("context IDs/image mapping is incomplete")
    offsets = store["action_offsets"].tolist()
    if len(offsets) != images + 1 or offsets[0] != 0 or offsets[-1] != actions or \
            any(right <= left for left, right in zip(offsets, offsets[1:])):
        errors.append("action offsets are invalid")
    tensors = {key: value for key, value in store.items() if isinstance(value, torch.Tensor)}
    nonfinite = {key: int((~torch.isfinite(value)).sum()) for key, value in tensors.items()
                 if value.is_floating_point() and not torch.isfinite(value).all()}
    if nonfinite:
        errors.append(f"nonfinite feature values: {nonfinite}")
    for key in ("original_context_features", "paraphrase_context_features",
                "completion_context_features"):
        if tuple(store[key].shape) != (contexts, 768):
            errors.append(f"unexpected {key} shape")
        norms = store[key].norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4):
            errors.append(f"{key} is not unit normalized")
    if feature_dim != 2315:
        errors.append(f"unexpected candidate feature dimension: {feature_dim}")
    else:
        features = store["candidate_features"]
        family = features[:, 2310:2313]
        if not torch.allclose(family.sum(1), torch.ones(actions)) or \
                not torch.all((family == 0) | (family == 1)):
            errors.append("action-family one-hot block is invalid")
        for index, candidate_id in enumerate(store["candidate_ids"]):
            metadata = candidates[candidate_id]
            expected = {"grounded_entity": 0, "full_image": 1, "stop": 2}[
                metadata["view_family"]]
            if family[index, expected] != 1:
                errors.append(f"wrong family bit for {candidate_id}")
                break
            crop, full, phrase = (features[index, :768], features[index, 768:1536],
                                  features[index, 1536:2304])
            if not torch.isclose(full.norm(), torch.tensor(1.0), atol=2e-4):
                errors.append(f"non-unit full-image feature for {candidate_id}")
                break
            if metadata["view_family"] == "stop" and (crop.any() or phrase.any()):
                errors.append(f"STOP has crop/phrase content: {candidate_id}")
                break
            if metadata["view_family"] == "full_image" and (
                    not torch.allclose(crop, full, atol=1e-6) or phrase.any()):
                errors.append(f"full-image action blocks are invalid: {candidate_id}")
                break
            if metadata["view_family"] == "grounded_entity" and (
                    not torch.isclose(crop.norm(), torch.tensor(1.0), atol=2e-4) or
                    not torch.isclose(phrase.norm(), torch.tensor(1.0), atol=2e-4)):
                errors.append(f"entity crop/phrase is not unit normalized: {candidate_id}")
                break
        cost = features[:, 2314]
        if torch.any((cost < 0) | (cost > 1.000001)):
            errors.append("cost ratio is outside [0,1]")
    source_names = " ".join(runtime.get("files", {})).lower()
    forbidden = ("strict", "label", "complement_type", "judgment", "entailment")
    leakage_terms = [term for term in forbidden if term in source_names]
    if leakage_terms:
        errors.append(f"outcome-like sources present: {leakage_terms}")
    report = {
        "status": "pass" if not errors else "fail",
        "scope": "mechanical feature/leakage audit; no outcome-quality claim",
        "counts": {"images": images, "contexts": contexts, "actions": actions,
                   "candidate_feature_dim": feature_dim,
                   "min_actions_per_image": min(b - a for a, b in zip(offsets, offsets[1:])),
                   "max_actions_per_image": max(b - a for a, b in zip(offsets, offsets[1:]))},
        "nonfinite": nonfinite, "leakage_terms": leakage_terms, "errors": errors,
    }
    (args.store_output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("R3-v2 feature store audit failed")


if __name__ == "__main__":
    main()
