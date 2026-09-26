"""Provisional crop-only support judgments for a small direction screen."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.observe import (DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl,
                                 crop_candidate, read_jsonl)


LABELS = ("SUPPORTED", "PARTIAL", "INVALID", "UNINFORMATIVE")
PROMPT = """Candidate observation: {observation}
Judge only against the supplied crop. SUPPORTED means every concrete claim is directly visible.
PARTIAL means at least one concrete claim is visible but another is unsupported. INVALID means no
concrete claim is supported or it relies on inference. UNINFORMATIVE means it states no recognizable
concrete fact. Reply with exactly SUPPORTED, PARTIAL, INVALID, or UNINFORMATIVE."""


def parse_label(text):
    value = text.strip().upper().strip(".:-").replace(" ", "_")
    if value == "SUPPORTIVE":
        value = "SUPPORTED"
    return value if value in LABELS else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--context-proposal-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    proposal_ids = sorted(row["staging_image_id"] for row in read_jsonl(
        args.context_proposal_output / "context_proposals.jsonl"))[:args.images]
    candidates = [row for row in read_jsonl(args.staging / "automatic_candidates.jsonl")
                  if row["staging_image_id"] in proposal_ids and row["kind"] != "stop"]
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    sources = [args.staging / "staging_images.jsonl",
               args.staging / "automatic_candidates.jsonl",
               args.observer_output / "observations.jsonl"]
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "prompt": PROMPT,
               "images": args.images, "batch_size": args.batch_size,
               "implementation": "provisional-visual-support-v1"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "visual_support.jsonl", args.output / "runtime.json"
    existing = {row["candidate_id"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different visual-support screen")
    for row in existing.values():
        label = parse_label(row.get("raw_output", ""))
        if label:
            row.update({"label": label, "status": "ok"})
    pending = []
    for candidate in candidates:
        observation = observations[candidate["candidate_id"]]["observed_text"]
        if candidate["candidate_id"] in existing:
            continue
        if observation == "NO_VISIBLE_FACT":
            existing[candidate["candidate_id"]] = {
                "staging_image_id": candidate["staging_image_id"],
                "candidate_id": candidate["candidate_id"], "label": "UNINFORMATIVE",
                "raw_output": "deterministic:NO_VISIBLE_FACT", "status": "ok"}
        else:
            pending.append((candidate, observation))
    metadata = {"scope": "provisional same-model crop support; not independent or human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "images": len(proposal_ids), "expected": len(candidates), "completed": len(existing),
                "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    if not pending and old_metadata:
        metadata = {**metadata, **old_metadata, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                                   local_files_only=True, use_fast=False)
        processor.tokenizer.padding_side = "left"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            crops, texts = [], []
            for candidate, observation in chunk:
                crops.append(crop_candidate(args.staging, images[candidate["staging_image_id"]],
                                            candidate))
                prompt = PROMPT.format(observation=observation)
                conversation = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, images=crops, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for (candidate, _), raw in zip(chunk, decoded):
                label = parse_label(raw)
                existing[candidate["candidate_id"]] = {
                    "staging_image_id": candidate["staging_image_id"],
                    "candidate_id": candidate["candidate_id"], "label": label,
                    "raw_output": raw.strip(), "status": "ok" if label else "parse_failed"}
            for crop in crops:
                crop.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"visual support {len(existing)}/{len(candidates)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if len(existing) != len(candidates) or any(row["status"] != "ok" for row in existing.values()):
        raise RuntimeError("visual-support cache incomplete or unparsed")
    print(json.dumps({"status": "provisional_complete", "counts": {
        label: sum(row["label"] == label for row in existing.values()) for label in LABELS}}, indent=2))


if __name__ == "__main__":
    main()
