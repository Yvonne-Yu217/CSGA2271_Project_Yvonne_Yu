"""Frozen full-image prompted planner baseline; never reads gold labels."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.observe import DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl, read_jsonl


PROMPT = """An existing image description is: {caption}
Choose exactly one observation that is most likely to reveal a correct visible fact not already
covered by that description. Regions use pixel boxes [left,top,right,bottom]. Choose STOP if the
description is already sufficient. Options:
{options}
Reply with only CHOICE=<integer>."""

MONTAGE_PROMPT = """An existing image description is: {caption}
The supplied montage contains numbered candidate observations. Choose exactly one candidate most
likely to reveal a correct visible fact not already covered by the description. Choose {stop_index}
for STOP if no additional observation is useful. Reply with only CHOICE=<integer>."""


def parse_choice(raw, candidates):
    match = re.search(r"(?:CHOICE\s*=\s*)?(\d+)\s*$", raw.strip(), re.I)
    if match:
        index = int(match.group(1))
        if 0 <= index < len(candidates):
            return index
    if raw.strip().upper() == "STOP":
        return next((index for index, row in enumerate(candidates) if row["kind"] == "stop"), -1)
    return -1


def candidate_montage(image_path, candidates, tile=224):
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        views = []
        indices = []
        for index, candidate in enumerate(candidates):
            if candidate["kind"] == "stop":
                continue
            crops = [source.crop(box) for box in candidate["boxes"]]
            if len(crops) == 1:
                view = crops[0]
            else:
                width = sum(crop.width for crop in crops)
                view = Image.new("RGB", (width, max(crop.height for crop in crops)), "white")
                x = 0
                for crop in crops:
                    view.paste(crop, (x, 0))
                    x += crop.width
            view.thumbnail((tile, tile))
            canvas = Image.new("RGB", (tile, tile), "white")
            canvas.paste(view, ((tile - view.width) // 2, (tile - view.height) // 2))
            ImageDraw.Draw(canvas).rectangle((0, 0, 42, 24), fill="black")
            ImageDraw.Draw(canvas).text((4, 4), str(index), fill="white")
            views.append(canvas)
            indices.append(index)
    columns = 4
    montage = Image.new("RGB", (columns * tile, ((len(views) + columns - 1) // columns) * tile),
                        "white")
    for position, view in enumerate(views):
        montage.paste(view, ((position % columns) * tile, (position // columns) * tile))
    return montage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--visualization", choices=("coordinates", "montage"),
                        default="coordinates")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = read_jsonl(args.staging / "natural_contexts.jsonl")
    candidates_by_image = {}
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        candidates_by_image.setdefault(row["staging_image_id"], []).append(row)
    for image_id, row in images.items():
        path = args.staging / row["image_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {image_id}")
    prompt_template = PROMPT if args.visualization == "coordinates" else MONTAGE_PROMPT
    payload = {
        "staging": {name: hashlib.sha256((args.staging / name).read_bytes()).hexdigest()
                    for name in ("staging_images.jsonl", "natural_contexts.jsonl",
                                 "automatic_candidates.jsonl")},
        "model": args.model, "revision": args.revision, "prompt": prompt_template,
        "batch_size": args.batch_size, "implementation": "qwen-planner-v1",
    }
    if args.visualization != "coordinates":
        payload["visualization"] = args.visualization
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "choices.jsonl", args.output / "runtime.json"
    existing = {row["context_id"]: row for row in read_jsonl(rows_path)}
    old = None
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError("output belongs to a different planner run")
    # Parser improvements are deterministic and reuse raw cached model output.
    contexts_by_id = {row["context_id"]: row for row in contexts}
    for context_id, row in existing.items():
        context = contexts_by_id[context_id]
        candidates = sorted(candidates_by_image[context["staging_image_id"]],
                            key=lambda candidate: candidate["candidate_id"])
        index = parse_choice(row["raw_output"], candidates)
        row.update({"candidate_id": candidates[index]["candidate_id"] if index >= 0 else None,
                    "choice_index": index if index >= 0 else None,
                    "status": "ok" if index >= 0 else "parse_failed"})
    if existing:
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    pending = [row for row in contexts if row["context_id"] not in existing]
    metadata = {
        "scope": "frozen model-visible planner baseline; no outcome claim before E0",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        "visualization": args.visualization,
        "contexts": len(contexts), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    if not pending and old:
        metadata = {**metadata, **old, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not pending:
        print("planner cache complete")
        return
    load_started = time.time()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                               local_files_only=not args.allow_model_download,
                                               use_fast=False)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.allow_model_download,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    metadata["model_load_seconds"] = time.time() - load_started
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset:offset + args.batch_size]
        batch_images, texts, ordered_candidates = [], [], []
        for context in chunk:
            image_id = context["staging_image_id"]
            candidates = sorted(candidates_by_image[image_id], key=lambda row: row["candidate_id"])
            ordered_candidates.append(candidates)
            if args.visualization == "coordinates":
                options = "\n".join(
                    f"{index}: {'STOP' if row['kind'] == 'stop' else json.dumps(row['boxes'])}"
                    for index, row in enumerate(candidates))
                prompt = PROMPT.format(caption=context["initial_caption"], options=options)
                visual = Image.open(args.staging / images[image_id]["image_path"]).convert("RGB")
            else:
                stop_index = next(index for index, row in enumerate(candidates)
                                  if row["kind"] == "stop")
                prompt = MONTAGE_PROMPT.format(caption=context["initial_caption"],
                                               stop_index=stop_index)
                visual = candidate_montage(args.staging / images[image_id]["image_path"], candidates)
            conversation = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                       add_generation_prompt=True))
            batch_images.append(visual)
        inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt").to("cuda")
        batch_started = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        latency = round(1000 * (time.time() - batch_started) / len(chunk))
        for context, candidates, raw, output_ids in zip(chunk, ordered_candidates, decoded, trimmed):
            index = parse_choice(raw, candidates)
            valid = 0 <= index < len(candidates)
            existing[context["context_id"]] = {
                "context_id": context["context_id"], "staging_image_id": context["staging_image_id"],
                "candidate_id": candidates[index]["candidate_id"] if valid else None,
                "choice_index": index if valid else None, "raw_output": raw.strip(),
                "status": "ok" if valid else "parse_failed", "output_tokens": int(output_ids.numel()),
                "latency_ms": latency,
            }
        for image in batch_images:
            image.close()
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"planner {len(existing)}/{len(contexts)}", flush=True)
    metadata.update({"status": "complete", "completed": len(existing),
                     "elapsed_seconds": time.time() - started,
                     "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     "gpu_name": torch.cuda.get_device_name(0)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
