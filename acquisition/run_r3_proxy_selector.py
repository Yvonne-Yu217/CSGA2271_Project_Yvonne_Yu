"""Frozen-SigLIP proxy selector for the predeclared R3-v0 development test."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import atomic_jsonl, crop_candidate, read_jsonl


MODEL = "google/siglip-base-patch16-224"
REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SALT = "r3-proxy-v1"
RIDGE = 1.0


def split_for(image_id):
    bucket = int(hashlib.sha256(f"{SALT}:{image_id}".encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else ("val" if bucket < 8 else "proxy_test")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ridge(train_features, targets, eval_features):
    mean = train_features.mean(0, keepdim=True)
    scale = train_features.std(0, keepdim=True).clamp_min(1e-6)
    train = (train_features - mean) / scale
    evaluate = (eval_features - mean) / scale
    train = torch.cat([train, torch.ones(len(train), 1, device=train.device)], dim=1)
    evaluate = torch.cat([evaluate, torch.ones(len(evaluate), 1, device=evaluate.device)], dim=1)
    identity = torch.eye(train.shape[1], device=train.device)
    weights = torch.linalg.solve(train.T @ train + RIDGE * identity, train.T @ targets)
    return evaluate @ weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--strict-output", type=Path, required=True)
    parser.add_argument("--full-image-core-output", type=Path, required=True)
    parser.add_argument("--coordinate-planner", type=Path, required=True)
    parser.add_argument("--montage-planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    candidates = {row["candidate_id"]: row for row in read_jsonl(
        args.staging / "automatic_candidates.jsonl") if row["kind"] != "stop"}
    labels = read_jsonl(args.strict_output / "complement_types.jsonl")
    if len(labels) != 7000:
        raise RuntimeError("expected 7000 R2 strict labels")
    label_lookup = {(row["context_id"], row["candidate_id"]):
                    int(row["label"] == "MENTIONED_ENTITY_DETAIL") for row in labels}
    source_paths = [args.staging / "staging_images.jsonl",
                    args.staging / "natural_contexts.jsonl",
                    args.staging / "automatic_candidates.jsonl",
                    args.strict_output / "complement_types.jsonl"]
    split_counts = Counter(split_for(image_id) for image_id in images)
    payload = {
        "implementation": "r3-proxy-siglip-ridge-v1",
        "files": {str(path): digest(path) for path in source_paths},
        "model": MODEL, "revision": REVISION, "split_salt": SALT,
        "split_counts": dict(split_counts), "ridge_lambda": RIDGE,
        "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path, runtime_path = args.output / "features.pt", args.output / "runtime.json"
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "automated-label proxy development; not formal E3 or confirmation",
        "status": "running", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.time()
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("fingerprint") != fingerprint:
            raise RuntimeError("feature cache fingerprint mismatch")
    else:
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            dtype=torch.float16).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        candidate_ids = sorted(candidates)
        image_features = {}
        for offset in range(0, len(candidate_ids), args.batch_size):
            chunk = candidate_ids[offset:offset + args.batch_size]
            crops = [crop_candidate(args.staging, images[candidates[key]["staging_image_id"]],
                                    candidates[key]) for key in chunk]
            inputs = processor(images=crops, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                features = torch.nn.functional.normalize(
                    model.get_image_features(**inputs).float(), dim=1).cpu()
            image_features.update(zip(chunk, features))
            for crop in crops:
                crop.close()
            print(f"R3 crop features {min(offset + len(chunk), len(candidate_ids))}/"
                  f"{len(candidate_ids)}", flush=True)
        context_ids = sorted(contexts)
        text_batches = []
        for offset in range(0, len(context_ids), args.batch_size):
            chunk = context_ids[offset:offset + args.batch_size]
            inputs = processor(text=[contexts[key]["initial_caption"] for key in chunk],
                               padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                text_batches.append(torch.nn.functional.normalize(
                    model.get_text_features(**inputs).float(), dim=1).cpu())
            print(f"R3 text features {min(offset + len(chunk), len(context_ids))}/"
                  f"{len(context_ids)}", flush=True)
        cache = {
            "fingerprint": fingerprint, "candidate_ids": candidate_ids,
            "context_ids": context_ids,
            "image_features": torch.stack([image_features[key] for key in candidate_ids]),
            "text_features": torch.cat(text_batches),
        }
        torch.save(cache, cache_path)
        runtime["feature_peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    candidate_index = {key: i for i, key in enumerate(cache["candidate_ids"])}
    context_index = {key: i for i, key in enumerate(cache["context_ids"])}
    rows = []
    candidates_by_image = defaultdict(list)
    for candidate_id, candidate in candidates.items():
        candidates_by_image[candidate["staging_image_id"]].append(candidate_id)
    for context_id in sorted(contexts):
        image_id = contexts[context_id]["staging_image_id"]
        for candidate_id in sorted(candidates_by_image[image_id]):
            rows.append((context_id, image_id, candidate_id,
                         label_lookup[(context_id, candidate_id)]))
    image = torch.stack([cache["image_features"][candidate_index[row[2]]] for row in rows]).to("cuda")
    text = torch.stack([cache["text_features"][context_index[row[0]]] for row in rows]).to("cuda")
    targets = torch.tensor([1.0 if row[3] else -1.0 for row in rows], device="cuda")
    feature_sets = {
        "image_only": image,
        "interaction": image * text,
        "image_plus_interaction": torch.cat([image, image * text], dim=1),
    }
    train_indices = [i for i, row in enumerate(rows) if split_for(row[1]) == "train"]
    scores = {}
    for method, features in feature_sets.items():
        scores[method] = fit_ridge(features[train_indices], targets[train_indices], features).cpu()
        print(f"R3 fit {method}", flush=True)
    full_image = {row["context_id"]: row for row in read_jsonl(
        args.full_image_core_output / "complement_types.jsonl")}
    planners = {
        "coordinate_planner": {row["context_id"]: row for row in read_jsonl(
            args.coordinate_planner / "choices.jsonl")},
        "montage_planner": {row["context_id"]: row for row in read_jsonl(
            args.montage_planner / "choices.jsonl")},
    }
    report = {"status": "provisional_not_evidence", "warning": runtime["scope"],
              "split_counts": dict(split_counts), "splits": {}}
    for split in ("val", "proxy_test"):
        context_ids = sorted(context_id for context_id, context in contexts.items()
                             if split_for(context["staging_image_id"]) == split)
        clusters = [contexts[key]["staging_image_id"] for key in context_ids]
        vectors = {}
        for method in feature_sets:
            selected = []
            for context_id in context_ids:
                indices = [i for i, row in enumerate(rows) if row[0] == context_id]
                best = max(indices, key=lambda i: (float(scores[method][i]), rows[i][2]))
                selected.append(rows[best][3])
            vectors[method] = selected
        vectors["recognition_oracle"] = [max(
            label_lookup[(context_id, candidate_id)]
            for candidate_id in candidates_by_image[contexts[context_id]["staging_image_id"]])
            for context_id in context_ids]
        vectors["full_image_completion"] = [
            int(full_image[key]["label"] == "MENTIONED_ENTITY_DETAIL") for key in context_ids]
        for method, planner in planners.items():
            vectors[method] = [label_lookup.get((key, planner[key]["candidate_id"]), 0)
                               for key in context_ids]
        report["splits"][split] = {
            "contexts": len(context_ids), "images": len(set(clusters)),
            "methods": {method: bootstrap_cluster(values, clusters, samples=10000, seed=2271)
                        for method, values in vectors.items()},
            "interaction_minus_image_only": paired_bootstrap(
                vectors["interaction"], vectors["image_only"], clusters,
                samples=10000, seed=2271),
            "interaction_minus_full_image": paired_bootstrap(
                vectors["interaction"], vectors["full_image_completion"], clusters,
                samples=10000, seed=2271),
        }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({
        "status": "complete", "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
