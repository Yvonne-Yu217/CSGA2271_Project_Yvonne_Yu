"""Caption-aware region observer for the R1 geometry/context redesign.

This is development inference only. Generated text is never treated as gold.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.observe import (DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl,
                                 crop_candidate, read_jsonl)


IMPLEMENTATION = "conditioned-observer-r1-v2-context-id-key"
PROMPTS = {
    "conditioned": (
        "Existing image description: {caption}\n"
        "Add the single most useful concrete visible fact not already stated or entailed by the "
        "description. Prefer an attribute, action, state, or relation of a specific mentioned "
        "entity. Preserve instance identity and do not infer unseen context. {view_instruction} "
        "Reply with one sentence of at most 30 words and no heading. If no useful new visible "
        "fact can be added, reply exactly NO_NEW_FACT."
    ),
    "independent": (
        "State the single most useful concrete fact directly visible in the supplied view. "
        "Include an object attribute, action, state, or relation when clear. Do not infer identity, "
        "intent, place, or unseen context. {view_instruction} Reply with one sentence of at most "
        "30 words and no heading. If nothing is recognizable, reply exactly NO_VISIBLE_FACT."
    ),
}
VIEW_INSTRUCTIONS = {
    "region_only": "The image is a region cropped from a larger source image.",
    "global_plus_roi": (
        "The first image is the full source and the second is a magnified region from that same "
        "image; use the full image for identity/context and the region for evidence."
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_hash(context_mode, caption):
    value = caption if context_mode == "conditioned" else ""
    return hashlib.sha256(value.encode()).hexdigest()


def cache_key(image_sha, context, candidate, input_mode, prompt_hash, revision,
              max_new_tokens, max_pixels):
    payload = {
        "implementation": IMPLEMENTATION,
        "image_sha256": image_sha,
        "context_id": context["context_id"],
        "context_hash": context_hash(context["context_mode"], context["caption"]),
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": candidate["kind"],
        "boxes": candidate["boxes"],
        "input_mode": input_mode,
        "global_view": input_mode == "global_plus_roi",
        "model_revision": revision,
        "prompt_sha256": prompt_hash,
        "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "processor": {"max_pixels": max_pixels, "use_fast": False},
        "output_schema_revision": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def choose_contexts(context_rows, image_ids, per_image):
    grouped = defaultdict(list)
    for row in context_rows:
        if row["staging_image_id"] in image_ids:
            grouped[row["staging_image_id"]].append(row)
    chosen = []
    for image_id in sorted(image_ids):
        rows = sorted(grouped[image_id], key=lambda row: row["context_id"])
        if len(rows) < per_image:
            raise RuntimeError(f"{image_id} has only {len(rows)} contexts")
        chosen.extend(rows[:per_image])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-file", type=Path,
                        help="Candidate JSONL; defaults to STAGING/automatic_candidates.jsonl")
    parser.add_argument("--contexts-file", type=Path,
                        help="Context JSONL; defaults to STAGING/natural_contexts.jsonl")
    parser.add_argument("--input-mode", choices=sorted(VIEW_INSTRUCTIONS), required=True)
    parser.add_argument("--context-mode", choices=sorted(PROMPTS), default="conditioned")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--contexts-per-image", type=int, default=1)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if min(args.batch_size, args.max_new_tokens, args.contexts_per_image) < 1:
        parser.error("batch size, token limit, and contexts per image must be positive")
    if args.max_images < 0 or args.max_pixels < 1:
        parser.error("max-images must be nonnegative and max-pixels positive")

    candidates_path = args.candidates_file or args.staging / "automatic_candidates.jsonl"
    contexts_path = args.contexts_file or args.staging / "natural_contexts.jsonl"
    source_paths = [args.staging / "staging_images.jsonl", contexts_path, candidates_path]
    images = {row["staging_image_id"]: row for row in read_jsonl(source_paths[0])}
    image_ids = sorted(images)
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    chosen_contexts = choose_contexts(read_jsonl(source_paths[1]), set(image_ids),
                                      args.contexts_per_image)
    candidates_by_image = defaultdict(list)
    for row in read_jsonl(source_paths[2]):
        if row["staging_image_id"] in images and row["kind"] != "stop":
            candidates_by_image[row["staging_image_id"]].append(row)
    for image_id in image_ids:
        image_file = args.staging / images[image_id]["image_path"]
        if digest(image_file) != images[image_id]["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {image_id}")
        candidates_by_image[image_id].sort(key=lambda row: row["candidate_id"])

    work = []
    for context_row in chosen_contexts:
        context = {
            "context_id": context_row["context_id"],
            "context_mode": args.context_mode,
            "caption": context_row["initial_caption"],
        }
        for candidate in candidates_by_image[context_row["staging_image_id"]]:
            work.append((context_row["staging_image_id"], context, candidate))

    prompt_template = PROMPTS[args.context_mode]
    prompt_hash = hashlib.sha256(json.dumps({
        "template": prompt_template,
        "view_instruction": VIEW_INSTRUCTIONS[args.input_mode],
    }, sort_keys=True).encode()).hexdigest()
    run_payload = {
        "implementation": IMPLEMENTATION,
        "sources": {str(path): digest(path) for path in source_paths},
        "input_mode": args.input_mode, "context_mode": args.context_mode,
        "model": args.model, "revision": args.revision, "prompt_sha256": prompt_hash,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels, "max_images": args.max_images,
        "contexts_per_image": args.contexts_per_image,
        "selected_context_ids": [row["context_id"] for row in chosen_contexts],
    }
    fingerprint = hashlib.sha256(json.dumps(run_payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "observations.jsonl"
    runtime_path = args.output / "runtime.json"
    old_runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old_runtime and old_runtime.get("fingerprint") != fingerprint:
        raise RuntimeError("output directory belongs to a different run fingerprint")
    existing = {row["cache_key"]: row for row in read_jsonl(rows_path)}
    pending = []
    for image_id, context, candidate in work:
        key = cache_key(images[image_id]["image_sha256"], context, candidate,
                        args.input_mode, prompt_hash, args.revision,
                        args.max_new_tokens, args.max_pixels)
        if key not in existing:
            pending.append((key, image_id, context, candidate))

    runtime = {
        **run_payload, "fingerprint": fingerprint,
        "scope": "development generated output; not human ground truth",
        "status": "running" if pending else "complete",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "total_outputs": len(work), "completed_outputs": len(existing),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if not pending:
        print("conditioned observer cache complete", flush=True)
        return

    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision,
        local_files_only=not args.allow_model_download, use_fast=False,
        max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision,
        local_files_only=not args.allow_model_download, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset:offset + args.batch_size]
        all_images, texts, opened = [], [], []
        for _, image_id, context, candidate in chunk:
            source = Image.open(args.staging / images[image_id]["image_path"]).convert("RGB")
            crop = crop_candidate(args.staging, images[image_id], candidate)
            opened.extend([source, crop])
            supplied = [crop] if args.input_mode == "region_only" else [source, crop]
            all_images.extend(supplied)
            prompt = prompt_template.format(
                caption=context["caption"],
                view_instruction=VIEW_INSTRUCTIONS[args.input_mode])
            content = [{"type": "image"} for _ in supplied] + [{"type": "text", "text": prompt}]
            texts.append(processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True))
        inputs = processor(text=texts, images=all_images, padding=True,
                           return_tensors="pt").to("cuda")
        batch_started = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        latency_ms = 1000 * (time.time() - batch_started) / len(chunk)
        for row_index, ((key, image_id, context, candidate), output_ids, text) in enumerate(
                zip(chunk, trimmed, decoded)):
            existing[key] = {
                "cache_key": key, "staging_image_id": image_id,
                "context_id": context["context_id"], "candidate_id": candidate["candidate_id"],
                "input_mode": args.input_mode, "context_mode": args.context_mode,
                "prompt_sha256": prompt_hash, "model_revision": args.revision,
                "observed_text": text.strip(), "status": "ok" if text.strip() else "empty",
                "input_tokens": int(inputs.attention_mask[row_index].sum().item()),
                "output_tokens": int(output_ids.numel()), "latency_ms": round(latency_ms, 3),
            }
        for image in opened:
            image.close()
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        runtime.update({
            "completed_outputs": len(existing), "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "status": "running",
        })
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        print(f"conditioned observer {len(existing)}/{len(work)}", flush=True)
    failures = sum(row["status"] != "ok" for row in existing.values())
    if len(existing) != len(work):
        raise RuntimeError(f"cache cardinality mismatch: {len(existing)} != {len(work)}")
    runtime.update({
        "status": "complete" if not failures else "fail", "failures": failures,
        "completed_outputs": len(existing), "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "gpu_name": torch.cuda.get_device_name(0), "completed_at_unix": time.time(),
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"empty outputs: {failures}")


if __name__ == "__main__":
    main()
