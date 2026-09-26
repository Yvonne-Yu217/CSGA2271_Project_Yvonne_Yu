"""Run a fixed caption-independent crop observer with resumable caching."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.schema import observation_cache_key

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_PROMPT = (
    "Describe only concrete facts directly visible inside this crop. Include objects and, when "
    "visible, their attributes, actions, or relations. Do not infer identity, intent, place, or "
    "unseen context. Reply with one sentence of at most 30 words and no heading. If nothing is "
    "recognizable, reply exactly NO_VISIBLE_FACT."
)


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def run_fingerprint(staging, model, revision, prompt, max_new_tokens, batch_size, max_images):
    payload = {
        "staging_hashes": {
            name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
            for name in ("staging_images.jsonl", "automatic_candidates.jsonl")
        },
        "model": model, "revision": revision, "prompt": prompt,
        "max_new_tokens": max_new_tokens, "batch_size": batch_size, "max_images": max_images,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "implementation": "observer-v3-integrity-accounting",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def crop_candidate_from_source(source, candidate):
    """Crop/montage a candidate from an already decoded RGB source image."""
    crops = [source.crop(box) for box in candidate["boxes"]]
    if len(crops) == 1:
        return crops[0]
    target_height = max(image.height for image in crops)
    resized = []
    for image in crops:
        width = max(1, round(image.width * target_height / image.height))
        resized.append(image.resize((width, target_height)))
    canvas = Image.new("RGB", (sum(image.width for image in resized) + 4 * (len(resized) - 1),
                               target_height), "white")
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width + 4
    return canvas


def crop_candidate(staging, image_row, candidate):
    with Image.open(staging / image_row["image_path"]) as source:
        source = source.convert("RGB")
        return crop_candidate_from_source(source, candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA device is required for the fixed observer")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max-new-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(args.staging, args.model, args.revision, args.prompt,
                                  args.max_new_tokens, args.batch_size, args.max_images)
    metadata_path = args.output / "observer_metadata.json"
    rows_path = args.output / "observations.jsonl"
    old = None
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError("output contains a different observer/staging fingerprint")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    image_ids = sorted(images)
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    for image_id in image_ids:
        image_row = images[image_id]
        image_file = args.staging / image_row["image_path"]
        if hashlib.sha256(image_file.read_bytes()).hexdigest() != image_row["image_sha256"]:
            raise RuntimeError(f"staged image hash mismatch: {image_id}")
    candidates = [row for row in read_jsonl(args.staging / "automatic_candidates.jsonl")
                  if row["staging_image_id"] in image_ids]
    existing = {row["candidate_id"]: row for row in read_jsonl(rows_path)}
    prompt_hash = hashlib.sha256(args.prompt.encode()).hexdigest()
    for candidate in candidates:
        image_row = images[candidate["staging_image_id"]]
        expected = observation_cache_key(image_row["image_sha256"], candidate,
                                         args.revision, prompt_hash)
        if candidate["candidate_id"] in existing and existing[candidate["candidate_id"]]["cache_key"] != expected:
            raise RuntimeError(f"cache key mismatch for {candidate['candidate_id']}")
        if candidate["kind"] == "stop" and candidate["candidate_id"] not in existing:
            existing[candidate["candidate_id"]] = {
                "observation_id": "obs_" + candidate["candidate_id"],
                "staging_image_id": candidate["staging_image_id"],
                "candidate_id": candidate["candidate_id"], "observer_revision": args.revision,
                "prompt_hash": prompt_hash, "cache_key": expected, "observed_text": "",
                "output_tokens": 0, "latency_ms": 0, "status": "stop",
            }
    pending = [row for row in candidates
               if row["kind"] != "stop" and row["candidate_id"] not in existing]
    metadata = {
        "scope": "fixed crop observer cache; outputs require blind fact review before E1/E2",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt": args.prompt, "prompt_hash": prompt_hash, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "requested_images": len(image_ids),
        "total_candidates": len(candidates), "completed_candidates": len(existing),
        "status": "running" if pending else "complete",
        "started_at_unix": time.time(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    if not pending and old:
        metadata = {**metadata, **old, "completed_candidates": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if not pending:
        print("observer cache complete")
        return

    load_started = time.time()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                               local_files_only=not args.allow_model_download,
                                               use_fast=False)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.allow_model_download,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    metadata["model_load_seconds"] = time.time() - load_started
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    completed_before = len(existing)
    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset:offset + args.batch_size]
        crops = [crop_candidate(args.staging, images[row["staging_image_id"]], row)
                 for row in chunk]
        conversations = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt},
        ]} for _ in chunk]
        texts = [processor.apply_chat_template([conversation], tokenize=False,
                                               add_generation_prompt=True)
                 for conversation in conversations]
        inputs = processor(text=texts, images=crops, padding=True, return_tensors="pt").to("cuda")
        batch_started = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        batch_ms = round(1000 * (time.time() - batch_started) / len(chunk))
        for row_index, (candidate, source_ids, output_ids, text) in enumerate(
                zip(chunk, inputs.input_ids, trimmed, decoded)):
            image_row = images[candidate["staging_image_id"]]
            existing[candidate["candidate_id"]] = {
                "observation_id": "obs_" + candidate["candidate_id"],
                "staging_image_id": candidate["staging_image_id"],
                "candidate_id": candidate["candidate_id"], "observer_revision": args.revision,
                "prompt_hash": prompt_hash,
                "cache_key": observation_cache_key(image_row["image_sha256"], candidate,
                                                   args.revision, prompt_hash),
                "observed_text": text.strip(),
                "input_tokens": int(inputs.attention_mask[row_index].sum().item()),
                "padded_input_tokens": int(source_ids.numel()),
                "output_tokens": int(output_ids.numel()), "latency_ms": batch_ms, "status": "ok",
                "crop_pixels": sum((b[2] - b[0]) * (b[3] - b[1]) for b in candidate["boxes"]),
            }
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        metadata.update({
            "completed_candidates": len(existing), "status": "running",
            "elapsed_generation_seconds": time.time() - started,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        })
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"observer {min(offset + len(chunk), len(pending))}/{len(pending)}; "
              f"cached {len(existing)}/{len(candidates)}", flush=True)
    metadata.update({
        "status": "complete", "completed_candidates": len(existing),
        "newly_generated_candidates": len(existing) - completed_before,
        "elapsed_generation_seconds": time.time() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "gpu_name": torch.cuda.get_device_name(0),
        "completed_at_unix": time.time(),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
