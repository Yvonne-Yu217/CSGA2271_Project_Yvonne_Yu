"""Order-symmetrized A/B logit baseline on VQ-FocusAmbiguity train+val."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from run_frozen_baseline import (
    BASE_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    SWAPPED_PROMPT,
    grouped_interval,
    metrics,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    source_paths = [args.source / "train_gt.json", args.source / "val_gt.json"]
    rows = []
    for path in source_paths:
        split = path.name.removesuffix("_gt.json")
        rows.extend({**row, "official_split": split} for row in json.loads(path.read_text()))
    rows.sort(key=lambda row: row["id"])
    payload = {
        "source_sha256": {path.name: sha256(path) for path in source_paths},
        "model": args.model, "revision": args.revision,
        "base_prompt": BASE_PROMPT, "swapped_prompt": SWAPPED_PROMPT,
        "implementation": "symmetric-next-token-logit-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    runtime_path = args.output / "runtime.json"
    existing = {}
    if predictions_path.is_file():
        existing = {row["id"]: row for row in
                    (json.loads(line) for line in predictions_path.read_text().splitlines() if line)}
    old_runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old_runtime and old_runtime.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different symmetric-logit run")
    for row in rows:
        existing.setdefault(row["id"], {})
        existing[row["id"]].update({
            "id": row["id"], "split": row["official_split"],
            "file_name": row["file_name"], "question": row["question"],
            "label": row["label"], "data_source": row["attributes"]["data_source"],
        })
    runtime = {
        "status": "running", "scope": "train+val only; test locked",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "records": len(rows),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(predictions_path, [existing[key] for key in sorted(existing)])

    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        size={"longest_edge": 768})
    processor.tokenizer.padding_side = "left"
    code_ids = {}
    for code in ("A", "B"):
        encoded = processor.tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{code} is not a single token: {encoded}")
        code_ids[code] = encoded[0]
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    variants = {
        "base": (BASE_PROMPT, "A", "B"),
        "swapped": (SWAPPED_PROMPT, "B", "A"),
    }
    for mode in ("text_only", "image_question"):
        for variant, (prompt, ambiguous_code, unambiguous_code) in variants.items():
            key = f"{mode}_{variant}_margin"
            pending = [row for row in rows if key not in existing[row["id"]]]
            for offset in range(0, len(pending), args.batch_size):
                chunk = pending[offset:offset + args.batch_size]
                texts, images = [], []
                for row in chunk:
                    content = []
                    if mode == "image_question":
                        content.append({"type": "image"})
                        images.append(Image.open(args.images / row["file_name"]).convert("RGB"))
                    content.append({"type": "text", "text": prompt.format(question=row["question"])})
                    texts.append(processor.apply_chat_template(
                        [{"role": "user", "content": content}], add_generation_prompt=True))
                processor_args = {"text": texts, "padding": True, "return_tensors": "pt"}
                if mode == "image_question":
                    processor_args["images"] = images
                inputs = processor(**processor_args).to("cuda")
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, :].float()
                ambiguous_logits = logits[:, code_ids[ambiguous_code]].cpu()
                unambiguous_logits = logits[:, code_ids[unambiguous_code]].cpu()
                for row, ambiguous_logit, unambiguous_logit in zip(
                        chunk, ambiguous_logits, unambiguous_logits):
                    margin = float(ambiguous_logit - unambiguous_logit)
                    existing[row["id"]].update({
                        key: margin,
                        f"{mode}_{variant}_prediction": (
                            "ambiguous" if margin >= 0 else "unambiguous"),
                    })
                for image in images:
                    image.close()
                atomic_jsonl(predictions_path, [existing[index] for index in sorted(existing)])
                runtime.update({
                    "completed_for_current_pass": len(rows) - len(pending) +
                                                  min(offset + len(chunk), len(pending)),
                    "current_pass": f"{mode}:{variant}",
                    "elapsed_seconds": time.time() - started,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                })
                runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
                print(f"{mode} {variant} {runtime['completed_for_current_pass']}/{len(rows)}",
                      flush=True)
        for row in existing.values():
            margin = (row[f"{mode}_base_margin"] + row[f"{mode}_swapped_margin"]) / 2
            row[f"{mode}_symmetric_margin"] = margin
            row[f"{mode}_symmetric_prediction"] = (
                "ambiguous" if margin >= 0 else "unambiguous")
        atomic_jsonl(predictions_path, [existing[index] for index in sorted(existing)])
    result_rows = [existing[index] for index in sorted(existing)]
    report = {
        "status": "complete_train_val_preflight_not_test_evidence",
        "warning": "Next-token A/B logits; semantic margins averaged across code orders.",
        "model": args.model, "revision": args.revision, "records": len(result_rows),
        "token_ids": code_ids, "overall": {}, "by_source": {},
    }
    for mode in ("text_only", "image_question"):
        for variant in ("base", "swapped", "symmetric"):
            prediction_key = f"{mode}_{variant}_prediction"
            name = f"{mode}_{variant}"
            report["overall"][name] = metrics(result_rows, prediction_key)
            report["overall"][name]["balanced_accuracy_image_grouped_ci95"] = grouped_interval(
                result_rows,
                lambda sample, key=prediction_key: metrics(sample, key)["balanced_accuracy"])
        base_key = f"{mode}_base_prediction"
        swapped_key = f"{mode}_swapped_prediction"
        report["overall"][f"{mode}_code_order_agreement"] = sum(
            row[base_key] == row[swapped_key] for row in result_rows) / len(result_rows)
    for source in sorted({row["data_source"] for row in result_rows}):
        subset = [row for row in result_rows if row["data_source"] == source]
        report["by_source"][source] = metrics(
            subset, "image_question_symmetric_prediction")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({
        "status": "complete", "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "gpu_name": torch.cuda.get_device_name(0), "token_ids": code_ids,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
