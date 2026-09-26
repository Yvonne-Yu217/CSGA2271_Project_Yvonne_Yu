"""Fixed-hyperparameter SigLIP linear probes for focus-ambiguity learnability."""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


MODEL = "google/siglip-base-patch16-224"
REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
RIDGE_LAMBDA = 1.0
LABELS = ("ambiguous", "unambiguous")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification_metrics(rows, predictions):
    counts = Counter((row["label"], prediction) for row, prediction in zip(rows, predictions))
    recalls = {}
    for label in LABELS:
        denominator = sum(row["label"] == label for row in rows)
        recalls[label] = counts[(label, label)] / denominator if denominator else None
    return {
        "n": len(rows),
        "accuracy": sum(row["label"] == pred for row, pred in zip(rows, predictions)) / len(rows),
        "balanced_accuracy": sum(value for value in recalls.values() if value is not None) /
                             sum(value is not None for value in recalls.values()),
        "ambiguous_recall": recalls["ambiguous"],
        "confusion": {f"true_{truth}:pred_{pred}": counts[(truth, pred)]
                      for truth in LABELS for pred in LABELS},
    }


def fit_ridge(train_features, train_labels, eval_features):
    mean = train_features.mean(0, keepdim=True)
    scale = train_features.std(0, keepdim=True).clamp_min(1e-6)
    train = (train_features - mean) / scale
    evaluate = (eval_features - mean) / scale
    train = torch.cat([train, torch.ones(len(train), 1)], dim=1)
    evaluate = torch.cat([evaluate, torch.ones(len(evaluate), 1)], dim=1)
    identity = torch.eye(len(train), dtype=train.dtype)
    # Dual closed form avoids an underdetermined 1537-dimensional primal solve.
    weights = train.T @ torch.linalg.solve(
        train @ train.T + RIDGE_LAMBDA * identity, train_labels)
    return evaluate @ weights


def bootstrap(rows, predictions, samples=10000, seed=2271):
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sampled_rows = [rows[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]
        values.append(classification_metrics(sampled_rows, sampled_predictions)["balanced_accuracy"])
    values.sort()
    return [values[int(samples * 0.025)], values[int(samples * 0.975)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    split_rows = {}
    source_paths = {}
    for split in ("train", "val"):
        path = args.source / f"{split}_gt.json"
        source_paths[split] = path
        split_rows[split] = [{**row, "official_split": split}
                             for row in json.loads(path.read_text())]
    payload = {
        "source_sha256": {split: sha256(path) for split, path in source_paths.items()},
        "model": args.model, "revision": args.revision, "ridge_lambda": RIDGE_LAMBDA,
        "modalities": ["question", "image", "image_question"],
        "implementation": "siglip-fixed-ridge-probe-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "features.pt"
    runtime_path = args.output / "runtime.json"
    old_runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old_runtime and old_runtime.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different SigLIP probe")
    runtime = {
        "status": "running", "scope": "fixed probe on train and val only; test locked",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "ridge_lambda": RIDGE_LAMBDA, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    if old_runtime:
        runtime.update({key: value for key, value in old_runtime.items()
                        if key in {"elapsed_seconds", "peak_gpu_memory_mb", "gpu_name"}})
        runtime["status"] = "running"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.time()
    extracted_features = False
    if feature_path.is_file():
        cache = torch.load(feature_path, map_location="cpu", weights_only=True)
        if cache.get("fingerprint") != fingerprint:
            raise RuntimeError("feature cache fingerprint mismatch")
    else:
        extracted_features = True
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                                   local_files_only=True)
        model = AutoModel.from_pretrained(args.model, revision=args.revision, local_files_only=True,
                                          dtype=torch.float16).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        cache = {"fingerprint": fingerprint, "splits": {}}
        for split in ("train", "val"):
            image_features, text_features = [], []
            rows = split_rows[split]
            for offset in range(0, len(rows), args.batch_size):
                chunk = rows[offset:offset + args.batch_size]
                images = [Image.open(args.images / row["file_name"]).convert("RGB")
                          for row in chunk]
                image_inputs = processor(images=images, return_tensors="pt").to("cuda")
                text_inputs = processor(text=[row["question"] for row in chunk], padding=True,
                                        truncation=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    image_batch = model.get_image_features(**image_inputs).float()
                    text_batch = model.get_text_features(**text_inputs).float()
                    image_batch = torch.nn.functional.normalize(image_batch, dim=1)
                    text_batch = torch.nn.functional.normalize(text_batch, dim=1)
                image_features.append(image_batch.cpu())
                text_features.append(text_batch.cpu())
                for image in images:
                    image.close()
                print(f"{split} features {min(offset + len(chunk), len(rows))}/{len(rows)}",
                      flush=True)
            cache["splits"][split] = {
                "ids": torch.tensor([row["id"] for row in rows]),
                "image": torch.cat(image_features), "question": torch.cat(text_features),
            }
        torch.save(cache, feature_path)
        runtime["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    train_labels = torch.tensor([
        1.0 if row["label"] == "ambiguous" else -1.0 for row in split_rows["train"]
    ])
    feature_sets = {}
    for split in ("train", "val"):
        image = cache["splits"][split]["image"].float()
        question = cache["splits"][split]["question"].float()
        feature_sets[split] = {
            "question": question,
            "image": image,
            "image_question": torch.cat([image, question], dim=1),
        }
    report = {
        "status": "complete_train_to_val_preflight_not_test_evidence",
        "model": args.model, "revision": args.revision, "ridge_lambda": RIDGE_LAMBDA,
        "train_records": 70, "val_records": 70,
        "warning": "One fixed linear probe per modality; no hyperparameter selection and no test use.",
        "majority_unambiguous": {}, "probes": {},
    }
    for split in ("train", "val"):
        rows = split_rows[split]
        majority = ["unambiguous"] * len(rows)
        report["majority_unambiguous"][split] = classification_metrics(rows, majority)
    for modality in ("question", "image", "image_question"):
        scores = fit_ridge(feature_sets["train"][modality], train_labels,
                           feature_sets["val"][modality])
        predictions = ["ambiguous" if score >= 0 else "unambiguous" for score in scores]
        metric = classification_metrics(split_rows["val"], predictions)
        metric["balanced_accuracy_bootstrap_ci95"] = bootstrap(split_rows["val"], predictions)
        metric["by_source"] = {}
        for source in sorted({row["attributes"]["data_source"] for row in split_rows["val"]}):
            indices = [index for index, row in enumerate(split_rows["val"])
                       if row["attributes"]["data_source"] == source]
            metric["by_source"][source] = classification_metrics(
                [split_rows["val"][index] for index in indices],
                [predictions[index] for index in indices])
        report["probes"][modality] = metric
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({"status": "complete", "gpu_name": torch.cuda.get_device_name(0)})
    if extracted_features:
        runtime["elapsed_seconds"] = time.time() - started
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
