"""Independent-VLM crop support sensitivity check for observer outputs."""
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from acquisition.observe import atomic_jsonl, crop_candidate_from_source, read_jsonl
from acquisition.screen_visual_support import LABELS, PROMPT, parse_label


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_REVISION = "81cd9a775a4d644f2faf4e7becff4559b46b14c7"


def parse_independent_label(text):
    parsed = parse_label(text)
    if parsed:
        return parsed
    matches = {label for label in LABELS if re.search(rf"\b{label}\b", text.upper())}
    return next(iter(matches)) if len(matches) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--context-proposal-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    image_ids = sorted(row["staging_image_id"] for row in read_jsonl(
        args.context_proposal_output / "context_proposals.jsonl"))[:args.images]
    candidates = [row for row in read_jsonl(args.staging / "automatic_candidates.jsonl")
                  if row["staging_image_id"] in image_ids and row["kind"] != "stop"]
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    sources = [args.staging / "staging_images.jsonl",
               args.staging / "automatic_candidates.jsonl",
               args.observer_output / "observations.jsonl"]
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "prompt": PROMPT,
               "images": args.images, "batch_size": args.batch_size,
               "implementation": "independent-visual-support-v2-left-pad-768px"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "visual_support.jsonl", args.output / "runtime.json"
    existing = {row["candidate_id"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another independent visual-support run")
    for row in existing.values():
        label = parse_independent_label(row.get("raw_output", ""))
        if label:
            row.update({"label": label, "status": "ok"})
    pending = []
    for candidate in candidates:
        observation = observations[candidate["candidate_id"]]["observed_text"]
        if (candidate["candidate_id"] in existing and
                existing[candidate["candidate_id"]].get("status") == "ok"):
            continue
        if observation == "NO_VISIBLE_FACT":
            existing[candidate["candidate_id"]] = {
                "staging_image_id": candidate["staging_image_id"],
                "candidate_id": candidate["candidate_id"], "label": "UNINFORMATIVE",
                "raw_output": "deterministic:NO_VISIBLE_FACT", "status": "ok"}
        else:
            pending.append((candidate, observation))
    metadata = {"scope": "independent VLM crop support sensitivity; not human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "images": len(image_ids), "expected": len(candidates), "completed": len(existing),
                "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    if not pending and old_metadata:
        metadata = {**metadata, **old_metadata, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                                   local_files_only=not args.allow_model_download,
                                                   size={"longest_edge": 768})
        processor.tokenizer.padding_side = "left"
        model = AutoModelForVision2Seq.from_pretrained(
            args.model, revision=args.revision, local_files_only=not args.allow_model_download,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            crops, texts = [], []
            source_cache = {}
            for candidate, observation in chunk:
                image_id = candidate["staging_image_id"]
                if image_id not in source_cache:
                    source_cache[image_id] = Image.open(
                        args.staging / images[image_id]["image_path"]).convert("RGB")
                crops.append(crop_candidate_from_source(source_cache[image_id], candidate))
                conversation = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(observation=observation)}]}]
                texts.append(processor.apply_chat_template(conversation,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, images=crops, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            for (candidate, _), raw in zip(chunk, decoded):
                label = parse_independent_label(raw)
                existing[candidate["candidate_id"]] = {
                    "staging_image_id": candidate["staging_image_id"],
                    "candidate_id": candidate["candidate_id"], "label": label,
                    "raw_output": raw.strip(), "status": "ok" if label else "parse_failed"}
            for crop in crops:
                crop.close()
            for source in source_cache.values():
                source.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"independent visual {len(existing)}/{len(candidates)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = list(existing.values())
    failures = [row for row in rows if row["status"] != "ok"]
    report = {"status": "complete" if len(rows) == len(candidates) and not failures else "fail",
              "scope": "independent VLM sensitivity only; not human gold",
              "counts": dict(Counter(row.get("label") for row in rows)),
              "rows": len(rows), "expected": len(candidates),
              "parse_failures": Counter(row.get("raw_output") for row in failures).most_common(20)}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise RuntimeError("independent visual-support cache incomplete or unparsed")


if __name__ == "__main__":
    main()
