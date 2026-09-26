"""Build the label-free frozen feature store for gated R3-v2 training."""
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from acquisition.observe import read_jsonl
from acquisition.run_r3_proxy_selector import MODEL, REVISION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_texts(texts, processor, model, batch_size):
    batches = []
    for offset in range(0, len(texts), batch_size):
        chunk = texts[offset:offset + batch_size]
        inputs = processor(text=chunk, padding=True, truncation=True,
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            batches.append(torch.nn.functional.normalize(
                model.get_text_features(**inputs).float(), dim=1).cpu())
        print(f"R3-v2 text features {min(offset + len(chunk), len(texts))}/{len(texts)}",
              flush=True)
    return torch.cat(batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--grid-cache", type=Path, required=True)
    parser.add_argument("--grounded-cache", type=Path, required=True)
    parser.add_argument("--grounded-candidates", type=Path, required=True)
    parser.add_argument("--paraphrase-contexts", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    paths = {
        "images": args.staging / "staging_images.jsonl",
        "contexts": args.staging / "natural_contexts.jsonl",
        "grid_candidates": args.staging / "automatic_candidates.jsonl",
        "grounded_candidates": args.grounded_candidates,
        "paraphrases": args.paraphrase_contexts,
        "completions": args.completion_output / "completions.jsonl",
        "grid_cache": args.grid_cache,
        "grounded_cache": args.grounded_cache,
    }
    payload = {
        "implementation": "r3-v2-label-free-feature-store-v1",
        "files": {key: digest(path) for key, path in paths.items()},
        "model": MODEL, "revision": REVISION, "batch_size": args.batch_size,
        "feature_spec": ["crop_siglip", "full_siglip", "phrase_siglip",
                         "geometry_6", "family_onehot_3", "proposal_score", "cost_ratio"],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    store_path, runtime_path = args.output / "feature_store.pt", args.output / "runtime.json"
    if store_path.is_file() or runtime_path.is_file():
        raise RuntimeError("refuse to overwrite an existing R3-v2 feature store")
    runtime = {**payload, "fingerprint": fingerprint, "status": "running",
               "scope": "label-free frozen features; no action outcomes or training",
               "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.time()
    images = {row["staging_image_id"]: row for row in read_jsonl(paths["images"])}
    contexts = {row["context_id"]: row for row in read_jsonl(paths["contexts"])}
    paraphrases = {row["context_id"]: row for row in read_jsonl(paths["paraphrases"])}
    completions = {row["context_id"]: row for row in read_jsonl(paths["completions"])}
    if not (set(contexts) == set(paraphrases) == set(completions)):
        raise RuntimeError("context, paraphrase, and completion IDs differ")
    grid_candidates = [row for row in read_jsonl(paths["grid_candidates"])
                       if row["kind"] != "stop"]
    grounded_candidates = read_jsonl(paths["grounded_candidates"])
    actions_by_image = defaultdict(list)
    for row in grounded_candidates:
        if row.get("view_family") in {"grounded_entity", "full_image", "stop"}:
            actions_by_image[row["staging_image_id"]].append(row)
    if set(actions_by_image) != set(images):
        raise RuntimeError("grounded action inventory does not cover every image")
    for image_id, rows in actions_by_image.items():
        if sum(row.get("view_family") == "full_image" for row in rows) != 1 or \
                sum(row.get("view_family") == "stop" for row in rows) != 1:
            raise RuntimeError(f"{image_id} lacks exactly one full-image/STOP action")

    grid = torch.load(args.grid_cache, map_location="cpu", weights_only=True)
    grounded = torch.load(args.grounded_cache, map_location="cpu", weights_only=True)
    grid_index = {key: i for i, key in enumerate(grid["candidate_ids"])}
    grounded_index = {key: i for i, key in enumerate(grounded["candidate_ids"])}
    context_ids = sorted(contexts)
    if context_ids != list(grid["context_ids"]) or context_ids != list(grounded["context_ids"]):
        raise RuntimeError("cached context order differs from canonical order")
    if not torch.allclose(grid["text_features"], grounded["text_features"], atol=1e-5):
        raise RuntimeError("original caption features differ across frozen caches")
    largest_grid = {}
    for candidate in grid_candidates:
        image_id = candidate["staging_image_id"]
        current = largest_grid.get(image_id)
        if current is None or candidate["pixel_cost"] > current["pixel_cost"]:
            largest_grid[image_id] = candidate
    if any(row["candidate_id"] not in grid_index for row in largest_grid.values()):
        raise RuntimeError("full-image grid candidates are absent from cache")

    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.float16).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    paraphrase_features = encode_texts(
        [paraphrases[key]["initial_caption"] for key in context_ids],
        processor, model, args.batch_size)
    completion_features = encode_texts(
        [completions[key]["completion"] for key in context_ids],
        processor, model, args.batch_size)

    candidate_ids, candidate_image_ids, candidate_features = [], [], []
    action_offsets = [0]
    for image_id in sorted(images):
        width, height = images[image_id]["width"], images[image_id]["height"]
        image_area = width * height
        full_feature = grid["image_features"][grid_index[largest_grid[image_id]["candidate_id"]]]
        rows = sorted(actions_by_image[image_id], key=lambda row: (
            {"grounded_entity": 0, "full_image": 1, "stop": 2}[row["view_family"]],
            row["candidate_id"]))
        for row in rows:
            family = row["view_family"]
            zeros = torch.zeros_like(full_feature)
            if family == "grounded_entity":
                index = grounded_index[row["candidate_id"]]
                crop_feature = grounded["image_features"][index]
                phrase_feature = grounded["label_features"][index]
            elif family == "full_image":
                crop_feature, phrase_feature = full_feature, zeros
            else:
                crop_feature, phrase_feature = zeros, zeros
            boxes = row.get("boxes") or []
            if boxes:
                x1 = min(box[0] for box in boxes) / width
                y1 = min(box[1] for box in boxes) / height
                x2 = max(box[2] for box in boxes) / width
                y2 = max(box[3] for box in boxes) / height
                geometry = [x1, y1, x2, y2, max(0, x2 - x1) * max(0, y2 - y1),
                            len(boxes) / 2]
            else:
                geometry = [0.0] * 6
            family_vector = [float(family == name)
                             for name in ("grounded_entity", "full_image", "stop")]
            scalars = torch.tensor(
                geometry + family_vector + [float(row.get("proposal_score") or 0.0),
                                             row.get("pixel_cost", 0) / image_area],
                dtype=torch.float32)
            candidate_features.append(torch.cat(
                [crop_feature.float(), full_feature.float(), phrase_feature.float(), scalars]))
            candidate_ids.append(row["candidate_id"])
            candidate_image_ids.append(image_id)
        action_offsets.append(len(candidate_ids))
    store = {
        "fingerprint": fingerprint, "model": MODEL, "revision": REVISION,
        "context_ids": context_ids,
        "context_image_ids": [contexts[key]["staging_image_id"] for key in context_ids],
        "original_context_features": grid["text_features"].float(),
        "paraphrase_context_features": paraphrase_features.float(),
        "completion_context_features": completion_features.float(),
        "candidate_ids": candidate_ids, "candidate_image_ids": candidate_image_ids,
        "candidate_features": torch.stack(candidate_features),
        "image_ids": sorted(images), "action_offsets": torch.tensor(action_offsets),
        "feature_blocks": payload["feature_spec"],
    }
    torch.save(store, store_path)
    runtime.update({
        "status": "complete", "contexts": len(context_ids), "images": len(images),
        "actions": len(candidate_ids), "candidate_feature_dim": store["candidate_features"].shape[1],
        "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "store_sha256": digest(store_path),
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
