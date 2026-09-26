"""Run frozen text-only and image+question ambiguity baselines on train+val only."""
import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_REVISION = "81cd9a775a4d644f2faf4e7becff4559b46b14c7"
BASE_PROMPT = """This is a classification task. Do NOT answer the embedded visual question.
Classify whether the embedded question has a unique intended visual focus:
A = two or more distinct image regions are plausible referents (ambiguous focus)
B = exactly one image region is the uniquely intended referent (unambiguous focus)
Judge referent uniqueness, not whether the answer is uncertain.
Embedded question, which you must NOT answer: <{question}>
Classification code (reply with only A or B):"""
SWAPPED_PROMPT = """This is a classification task. Do NOT answer the embedded visual question.
Classify whether the embedded question has a unique intended visual focus:
A = exactly one image region is the uniquely intended referent (unambiguous focus)
B = two or more distinct image regions are plausible referents (ambiguous focus)
Judge referent uniqueness, not whether the answer is uncertain.
Embedded question, which you must NOT answer: <{question}>
Classification code (reply with only A or B):"""
PROMPT_VARIANTS = {
    "base": (BASE_PROMPT, {"a": "ambiguous", "b": "unambiguous"}),
    "swapped": (SWAPPED_PROMPT, {"a": "unambiguous", "b": "ambiguous"}),
}
LABELS = ("ambiguous", "unambiguous")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_label(text, code_labels):
    normalized = text.strip().lower().strip(" .,:;!*-_")
    if normalized in code_labels:
        return code_labels[normalized]
    if normalized in LABELS:
        return normalized
    matches = {label for label in LABELS if re.search(rf"\b{label}\b", normalized)}
    return next(iter(matches)) if len(matches) == 1 else None


def metrics(rows, prediction_key):
    counts = Counter((row["label"], row[prediction_key]) for row in rows)
    total = len(rows)
    accuracy = sum(row["label"] == row[prediction_key] for row in rows) / total
    recalls = {}
    for label in LABELS:
        subset = [row for row in rows if row["label"] == label]
        recalls[label] = (sum(row[prediction_key] == label for row in subset) / len(subset)
                          if subset else None)
    predicted_ambiguous = sum(row[prediction_key] == "ambiguous" for row in rows)
    true_positive = counts[("ambiguous", "ambiguous")]
    actual_ambiguous = sum(row["label"] == "ambiguous" for row in rows)
    precision = true_positive / predicted_ambiguous if predicted_ambiguous else 0.0
    recall = true_positive / actual_ambiguous if actual_ambiguous else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": total,
        "accuracy": accuracy,
        "balanced_accuracy": (sum(value for value in recalls.values() if value is not None) /
                              sum(value is not None for value in recalls.values())),
        "ambiguous_precision": precision,
        "ambiguous_recall": recall,
        "ambiguous_f1": f1,
        "confusion": {f"true_{truth}:pred_{prediction}": counts[(truth, prediction)]
                      for truth in LABELS for prediction in LABELS},
    }


def grouped_interval(rows, statistic, samples=10000, seed=2271):
    groups = defaultdict(list)
    for row in rows:
        groups[row["file_name"]].append(row)
    names = sorted(groups)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = []
        for _ in names:
            sampled.extend(groups[rng.choice(names)])
        estimates.append(statistic(sampled))
    estimates.sort()
    return {
        "estimate": statistic(rows),
        "ci95": [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]],
        "resampling_unit": "image filename",
        "bootstrap_samples": samples,
    }


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Deterministic prefix for format smoke tests; zero runs all 140")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="base")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    prompt, code_labels = PROMPT_VARIANTS[args.prompt_variant]
    source_paths = [args.source / "train_gt.json", args.source / "val_gt.json"]
    records = []
    for path in source_paths:
        official_split = path.name.removesuffix("_gt.json")
        for source_row in json.loads(path.read_text()):
            records.append({**source_row, "_official_split": official_split})
    records.sort(key=lambda row: row["id"])
    if args.limit:
        records = records[:args.limit]
    payload = {
        "source_sha256": {path.name: sha256(path) for path in source_paths},
        "model": args.model,
        "revision": args.revision,
        "prompt": prompt, "prompt_variant": args.prompt_variant,
        "modes": ["text_only", "image_question"],
        "implementation": "frozen-focus-ambiguity-baseline-v1",
        "limit": args.limit,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "predictions.jsonl"
    runtime_path = args.output / "runtime.json"
    existing = {}
    if rows_path.is_file():
        existing = {row["id"]: row for row in
                    (json.loads(line) for line in rows_path.read_text().splitlines() if line)}
    old_runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old_runtime and old_runtime.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different frozen baseline")
    for row in records:
        existing.setdefault(row["id"], {})
        existing[row["id"]].update({
            "id": row["id"], "split": row["_official_split"], "file_name": row["file_name"],
            "question": row["question"], "label": row["label"],
            "data_source": row["attributes"]["data_source"],
        })
    runtime = {
        "status": "running", "scope": "official train+val only; test remains locked",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "records": len(records),
    }
    if old_runtime:
        runtime.update({key: value for key, value in old_runtime.items()
                        if key.endswith("_elapsed_seconds") or key == "peak_gpu_memory_mb"})
        runtime["status"] = "running"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])

    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.allow_model_download,
        size={"longest_edge": 768})
    processor.tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.allow_model_download,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats()
    total_started = time.time()
    prior_total_elapsed = float(runtime.get("total_elapsed_seconds", 0.0))
    ran_inference = False
    for mode in ("text_only", "image_question"):
        pending = [row for row in records if existing[row["id"]].get(f"{mode}_status") != "ok"]
        mode_started = time.time()
        prior_mode_elapsed = float(runtime.get(f"{mode}_elapsed_seconds", 0.0))
        ran_inference = ran_inference or bool(pending)
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            texts, batch_images = [], []
            for row in chunk:
                content = []
                if mode == "image_question":
                    content.append({"type": "image"})
                    batch_images.append(Image.open(args.images / row["file_name"]).convert("RGB"))
                content.append({"type": "text", "text": prompt.format(question=row["question"])})
                conversation = [{"role": "user", "content": content}]
                texts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))
            processor_args = {"text": texts, "padding": True, "return_tensors": "pt"}
            if mode == "image_question":
                processor_args["images"] = batch_images
            inputs = processor(**processor_args).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                           use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            for row, raw in zip(chunk, decoded):
                prediction = parse_label(raw, code_labels)
                existing[row["id"]].update({
                    f"{mode}_prediction": prediction,
                    f"{mode}_raw": raw.strip(),
                    f"{mode}_status": "ok" if prediction else "parse_failed",
                })
            for image in batch_images:
                image.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            completed = sum(row.get(f"{mode}_status") == "ok" for row in existing.values())
            runtime.update({
                f"{mode}_completed": completed,
                f"{mode}_elapsed_seconds": prior_mode_elapsed + time.time() - mode_started,
                "total_elapsed_seconds": prior_total_elapsed + time.time() - total_started,
                "peak_gpu_memory_mb": max(float(runtime.get("peak_gpu_memory_mb", 0.0)),
                                          torch.cuda.max_memory_allocated() / 1024 ** 2),
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"{mode} {completed}/{len(records)}", flush=True)
    failures = [row for row in existing.values()
                if any(row.get(f"{mode}_status") != "ok"
                       for mode in ("text_only", "image_question"))]
    if failures:
        raise RuntimeError(f"unparsed predictions: {len(failures)}")
    rows = [existing[key] for key in sorted(existing)]
    report = {
        "status": "complete_train_val_preflight_not_test_evidence",
        "warning": "Frozen zero-shot preflight on 140 public train+val records; no test inference.",
        "model": args.model, "revision": args.revision, "records": len(rows),
        "prompt_variant": args.prompt_variant,
        "unique_images": len({row["file_name"] for row in rows}),
        "overall": {}, "by_source": {}, "by_split": {},
    }
    for mode in ("text_only", "image_question"):
        key = f"{mode}_prediction"
        report["overall"][mode] = metrics(rows, key)
        report["overall"][mode]["accuracy_image_grouped_ci95"] = grouped_interval(
            rows, lambda sample, key=key: metrics(sample, key)["accuracy"])
        report["overall"][mode]["balanced_accuracy_image_grouped_ci95"] = grouped_interval(
            rows, lambda sample, key=key: metrics(sample, key)["balanced_accuracy"])
        for source in sorted({row["data_source"] for row in rows}):
            report["by_source"].setdefault(source, {})[mode] = metrics(
                [row for row in rows if row["data_source"] == source], key)
        for split in ("train", "val"):
            report["by_split"].setdefault(split, {})[mode] = metrics(
                [row for row in rows if row["split"] == split], key)
    delta = lambda sample: (metrics(sample, "image_question_prediction")["balanced_accuracy"] -
                            metrics(sample, "text_only_prediction")["balanced_accuracy"])
    report["image_minus_text_balanced_accuracy"] = grouped_interval(rows, delta)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({"status": "complete", "gpu_name": torch.cuda.get_device_name(0)})
    if ran_inference:
        runtime["total_elapsed_seconds"] = prior_total_elapsed + time.time() - total_started
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
