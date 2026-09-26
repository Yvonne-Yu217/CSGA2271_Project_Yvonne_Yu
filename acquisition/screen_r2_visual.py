"""Context-keyed independent visual support screen for multi-caption R2 outputs."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from acquisition.observe import (atomic_jsonl, crop_candidate_from_source, read_jsonl)
from acquisition.retry_visual_parse import CODES, RETRY_PROMPT
from acquisition.screen_independent_visual import (DEFAULT_MODEL, DEFAULT_REVISION,
                                                   parse_independent_label)
from acquisition.screen_visual_support import PROMPT


IMPLEMENTATION = "r2-context-keyed-independent-visual-v1"


def generate_batches(pending, prompt_template, processor, model, staging, images,
                     candidates, batch_size, existing, coded=False):
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset:offset + batch_size]
        source_cache, crops, texts = {}, [], []
        for item in chunk:
            image_id = item["staging_image_id"]
            if image_id not in source_cache:
                source_cache[image_id] = Image.open(
                    staging / images[image_id]["image_path"]).convert("RGB")
            candidate = candidates[item["candidate_id"]]
            crops.append(crop_candidate_from_source(source_cache[image_id], candidate))
            prompt = prompt_template.format(observation=item["observed_text"])
            conversation = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(processor.apply_chat_template(conversation,
                                                       add_generation_prompt=True))
        inputs = processor(text=texts, images=crops, padding=True,
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=4 if coded else 16,
                                       do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        for item, raw in zip(chunk, decoded):
            if coded:
                code = raw.strip().upper().strip(".:- ")
                label = CODES.get(code)
            else:
                code = None
                label = parse_independent_label(raw)
            previous = existing.get(item["cache_key"], {})
            existing[item["cache_key"]] = {
                "cache_key": item["cache_key"],
                "staging_image_id": item["staging_image_id"],
                "context_id": item["context_id"], "candidate_id": item["candidate_id"],
                "label": label, "raw_output": raw.strip(),
                "status": "ok" if label else "parse_failed",
            }
            if coded:
                existing[item["cache_key"]].update({
                    "raw_output_initial": previous.get("raw_output"), "retry_code": code,
                    "retry_prompt_sha256": hashlib.sha256(RETRY_PROMPT.encode()).hexdigest(),
                })
        for crop in crops:
            crop.close()
        for source in source_cache.values():
            source.close()
        yield min(offset + len(chunk), len(pending))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    observations = sorted(read_jsonl(args.observer_output / "observations.jsonl"),
                          key=lambda row: (row["staging_image_id"], row["context_id"],
                                           row["candidate_id"]))
    if len(observations) != 7000 or any(row.get("status") != "ok" for row in observations):
        raise RuntimeError("expected 7000 complete R2 observations")
    source_paths = [args.staging / "staging_images.jsonl",
                    args.staging / "automatic_candidates.jsonl",
                    args.observer_output / "observations.jsonl"]
    payload = {
        "implementation": IMPLEMENTATION,
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths},
        "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "retry_prompt_sha256": hashlib.sha256(RETRY_PROMPT.encode()).hexdigest(),
        "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "visual_support.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different R2 visual fingerprint")
    existing = {row["cache_key"]: row for row in read_jsonl(rows_path)}
    for row in observations:
        if row["cache_key"] not in existing and row["observed_text"].strip().upper() in {
                "NO_NEW_FACT", "NO_VISIBLE_FACT"}:
            existing[row["cache_key"]] = {
                "cache_key": row["cache_key"], "staging_image_id": row["staging_image_id"],
                "context_id": row["context_id"], "candidate_id": row["candidate_id"],
                "label": "UNINFORMATIVE", "raw_output": "deterministic:no-fact",
                "status": "ok",
            }
    pending = [row for row in observations if row["cache_key"] not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "independent visual sensitivity; not human gold",
        "expected": len(observations), "completed": len(existing),
        "status": "running" if pending else "complete",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            size={"longest_edge": 768})
        processor.tokenizer.padding_side = "left"
        model = AutoModelForVision2Seq.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for completed in generate_batches(
                pending, PROMPT, processor, model, args.staging, images, candidates,
                args.batch_size, existing):
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({
                "completed": len(existing), "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"R2 visual {len(existing)}/{len(observations)}", flush=True)
        retry = [row for row in observations
                 if existing[row["cache_key"]].get("status") != "ok"]
        for _ in generate_batches(
                retry, RETRY_PROMPT, processor, model, args.staging, images, candidates,
                args.batch_size, existing, coded=True):
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            print(f"R2 visual strict retry {len(retry)}", flush=True)
    failures = [row for row in existing.values() if row.get("status") != "ok"]
    report = {
        "status": "complete" if len(existing) == len(observations) and not failures else "fail",
        "scope": "independent VLM sensitivity only; not human gold",
        "counts": dict(Counter(row.get("label") for row in existing.values())),
        "rows": len(existing), "expected": len(observations),
        "parse_failures": len(failures),
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    runtime.update({
        "status": report["status"], "completed": len(existing),
        "failures": len(failures), "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise RuntimeError("R2 visual screen incomplete")


if __name__ == "__main__":
    main()
