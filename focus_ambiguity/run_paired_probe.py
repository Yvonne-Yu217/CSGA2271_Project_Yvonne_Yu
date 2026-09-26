"""Same-image paired focus-ambiguity probe with a locked internal holdout."""
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


DEFAULT_MODEL = "google/siglip-base-patch16-224"
DEFAULT_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
RIDGE_LAMBDA = 1.0
SPLIT_SALT = "paired-focus-v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assigned_split(file_name):
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{file_name}".encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else ("val" if bucket < 8 else "holdout")


def prepare_pairs(source):
    excluded_images = set()
    for split in ("train", "val"):
        excluded_images.update(row["file_name"] for row in json.loads(
            (source / f"{split}_gt.json").read_text()))
    grouped = defaultdict(list)
    for row in json.loads((source / "test_gt.json").read_text()):
        grouped[row["file_name"]].append(row)
    pairs = []
    for file_name in sorted(grouped):
        rows = grouped[file_name]
        labels = Counter(row["label"] for row in rows)
        if (len(rows) == 2 and labels == {"ambiguous": 1, "unambiguous": 1}
                and file_name not in excluded_images
                and all(row["attributes"]["data_source"] == "paco" for row in rows)):
            pairs.append({
                "file_name": file_name,
                "split": assigned_split(file_name),
                "rows": sorted(rows, key=lambda row: row["id"]),
            })
    return pairs, excluded_images


def fit_ridge(train_features, train_targets, eval_features):
    mean = train_features.mean(0, keepdim=True)
    scale = train_features.std(0, keepdim=True).clamp_min(1e-6)
    train = (train_features - mean) / scale
    evaluate = (eval_features - mean) / scale
    train = torch.cat([train, torch.ones(len(train), 1, device=train.device)], dim=1)
    evaluate = torch.cat([evaluate, torch.ones(len(evaluate), 1, device=evaluate.device)], dim=1)
    if train.shape[1] <= train.shape[0]:
        identity = torch.eye(train.shape[1], device=train.device)
        weights = torch.linalg.solve(
            train.T @ train + RIDGE_LAMBDA * identity, train.T @ train_targets)
    else:
        identity = torch.eye(train.shape[0], device=train.device)
        weights = train.T @ torch.linalg.solve(
            train @ train.T + RIDGE_LAMBDA * identity, train_targets)
    return evaluate @ weights


def paired_metrics(rows, scores):
    by_image = defaultdict(list)
    correct = []
    for row, score in zip(rows, scores):
        predicted = "ambiguous" if score >= 0 else "unambiguous"
        correct.append(predicted == row["label"])
        by_image[row["file_name"]].append((row["label"], float(score), predicted))
    pair_correct, pair_rank = [], []
    for file_name, values in by_image.items():
        if len(values) != 2 or {value[0] for value in values} != {"ambiguous", "unambiguous"}:
            raise RuntimeError(f"invalid validation pair: {file_name}")
        pair_correct.append(all(label == prediction for label, _, prediction in values))
        ambiguous_score = next(score for label, score, _ in values if label == "ambiguous")
        unambiguous_score = next(score for label, score, _ in values if label == "unambiguous")
        pair_rank.append(1.0 if ambiguous_score > unambiguous_score else
                         (0.5 if ambiguous_score == unambiguous_score else 0.0))
    return {
        "records": len(rows), "pairs": len(by_image),
        "record_accuracy": sum(correct) / len(correct),
        "pair_accuracy": sum(pair_correct) / len(pair_correct),
        "pair_ranking_accuracy": sum(pair_rank) / len(pair_rank),
    }


def paired_bootstrap(rows, scores_by_method, samples=10000, seed=2271):
    indices_by_image = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_image[row["file_name"]].append(index)
    images = sorted(indices_by_image)
    rng = random.Random(seed)
    outcomes = {}
    for method, scores in scores_by_method.items():
        method_outcomes = {}
        for file_name in images:
            values = [(rows[index]["label"], float(scores[index]))
                      for index in indices_by_image[file_name]]
            ambiguous_score = next(score for label, score in values
                                   if label == "ambiguous")
            unambiguous_score = next(score for label, score in values
                                     if label == "unambiguous")
            predictions = [(label, "ambiguous" if score >= 0 else "unambiguous")
                           for label, score in values]
            method_outcomes[file_name] = {
                "pair_accuracy": float(all(label == prediction
                                           for label, prediction in predictions)),
                "pair_ranking_accuracy": (
                    1.0 if ambiguous_score > unambiguous_score else
                    (0.5 if ambiguous_score == unambiguous_score else 0.0)),
            }
        outcomes[method] = method_outcomes
    distributions = {method: {"pair_accuracy": [], "pair_ranking_accuracy": []}
                     for method in scores_by_method}
    distributions["interaction_minus_question"] = {
        "pair_accuracy": [], "pair_ranking_accuracy": []}
    for _ in range(samples):
        sampled_images = [rng.choice(images) for _ in images]
        sampled_metrics = {}
        for method in scores_by_method:
            sampled_metrics[method] = {
                metric: sum(outcomes[method][file_name][metric]
                            for file_name in sampled_images) / len(sampled_images)
                for metric in ("pair_accuracy", "pair_ranking_accuracy")
            }
            for metric in ("pair_accuracy", "pair_ranking_accuracy"):
                distributions[method][metric].append(sampled_metrics[method][metric])
        for metric in ("pair_accuracy", "pair_ranking_accuracy"):
            distributions["interaction_minus_question"][metric].append(
                sampled_metrics["interaction"][metric] - sampled_metrics["question"][metric])
    result = {}
    for method, metric_values in distributions.items():
        result[method] = {}
        for metric, values in metric_values.items():
            values.sort()
            result[method][metric] = [values[int(samples * 0.025)],
                                      values[int(samples * 0.975)]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    pairs, excluded_images = prepare_pairs(args.source)
    counts = Counter(pair["split"] for pair in pairs)
    source_paths = [args.source / f"{split}_gt.json" for split in ("train", "val", "test")]
    protocol = {
        "status": "frozen_before_model_inference",
        "split_salt": SPLIT_SALT,
        "split_rule": "sha256(salt:file_name) mod 10; 0-5 train, 6-7 val, 8-9 holdout",
        "pair_definition": "same PACO image, exactly one ambiguous and one unambiguous question",
        "excluded_official_train_val_images": len(excluded_images),
        "pairs": dict(sorted(counts.items())),
        "records": {split: 2 * count for split, count in sorted(counts.items())},
        "source_sha256": {path.name: sha256(path) for path in source_paths},
        "holdout_policy": "No holdout image is loaded or encoded during development.",
        "continue_rule": (
            "Interaction pair-ranking minus question-only >= 0.05 with paired-bootstrap "
            "95% CI lower bound > 0, replicated with CLIP and SigLIP."
        ),
        "stop_rule": "Stop if either encoder fails the continue rule.",
    }
    manifest_payload = [{"file_name": pair["file_name"], "split": pair["split"],
                         "ids": [row["id"] for row in pair["rows"]]} for pair in pairs]
    protocol["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("existing paired protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)
    if args.protocol_only:
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    development_pairs = [pair for pair in pairs if pair["split"] in {"train", "val"}]
    development_rows = [
        {**row, "paired_split": pair["split"]}
        for pair in development_pairs for row in pair["rows"]
    ]
    payload = {
        "protocol": protocol, "model": args.model, "revision": args.revision,
        "ridge_lambda": RIDGE_LAMBDA, "implementation": "paired-focus-probe-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    feature_path = args.output / "features.pt"
    runtime_path = args.output / "runtime.json"
    runtime = {
        "status": "running", "fingerprint": fingerprint, "model": args.model,
        "revision": args.revision, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "holdout_encoded": False,
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.time()
    if feature_path.is_file():
        cache = torch.load(feature_path, map_location="cpu", weights_only=True)
        if cache.get("fingerprint") != fingerprint:
            raise RuntimeError("feature cache fingerprint mismatch")
    else:
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, local_files_only=True)
        model = AutoModel.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            dtype=torch.float16).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        unique_images = sorted({row["file_name"] for row in development_rows})
        image_features = {}
        for offset in range(0, len(unique_images), args.batch_size):
            names = unique_images[offset:offset + args.batch_size]
            images = [Image.open(args.images / name).convert("RGB") for name in names]
            inputs = processor(images=images, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                features = torch.nn.functional.normalize(
                    model.get_image_features(**inputs).float(), dim=1).cpu()
            image_features.update(zip(names, features))
            for image in images:
                image.close()
            print(f"images {min(offset + len(names), len(unique_images))}/{len(unique_images)}",
                  flush=True)
        text_batches = []
        for offset in range(0, len(development_rows), args.batch_size):
            chunk = development_rows[offset:offset + args.batch_size]
            inputs = processor(text=[row["question"] for row in chunk], padding=True,
                               truncation=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                text_batches.append(torch.nn.functional.normalize(
                    model.get_text_features(**inputs).float(), dim=1).cpu())
            print(f"questions {min(offset + len(chunk), len(development_rows))}/"
                  f"{len(development_rows)}", flush=True)
        cache = {
            "fingerprint": fingerprint,
            "ids": torch.tensor([row["id"] for row in development_rows]),
            "image": torch.stack([image_features[row["file_name"]]
                                  for row in development_rows]),
            "question": torch.cat(text_batches),
        }
        torch.save(cache, feature_path)
        runtime["feature_peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    if cache["ids"].tolist() != [row["id"] for row in development_rows]:
        raise RuntimeError("feature row IDs differ from protocol")
    image = cache["image"].float().to("cuda")
    question = cache["question"].float().to("cuda")
    feature_sets = {
        "question": question,
        "image": image,
        "concat": torch.cat([image, question], dim=1),
        "interaction": image * question,
    }
    train_indices = [index for index, row in enumerate(development_rows)
                     if row["paired_split"] == "train"]
    val_indices = [index for index, row in enumerate(development_rows)
                   if row["paired_split"] == "val"]
    targets = torch.tensor([1.0 if row["label"] == "ambiguous" else -1.0
                            for row in development_rows], device="cuda")
    val_rows = [development_rows[index] for index in val_indices]
    scores_by_method = {}
    for method, features in feature_sets.items():
        scores = fit_ridge(features[train_indices], targets[train_indices],
                           features[val_indices]).cpu().tolist()
        scores_by_method[method] = scores
        print(f"fit {method}", flush=True)
    report = {
        "status": "complete_development_only_holdout_locked",
        "model": args.model, "revision": args.revision,
        "ridge_lambda": RIDGE_LAMBDA, "protocol": protocol,
        "methods": {method: paired_metrics(val_rows, scores)
                    for method, scores in scores_by_method.items()},
    }
    report["paired_bootstrap_ci95"] = paired_bootstrap(val_rows, scores_by_method)
    for metric in ("pair_accuracy", "pair_ranking_accuracy"):
        report["methods"]["interaction"][f"minus_question_{metric}"] = (
            report["methods"]["interaction"][metric] -
            report["methods"]["question"][metric]
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({
        "status": "complete", "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0), "holdout_encoded": False,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
