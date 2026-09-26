"""Generate review-only E2 context proposals from full images and natural captions."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.observe import DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl, read_jsonl


PROMPT = """The verified starting caption is: {caption}
Draft three captions for human review using only facts visibly supported by the image:
1. enriched: retain the starting facts and add one useful attribute or relation.
2. paraphrase: express the enriched caption with the same facts but different wording.
3. saturated: cover all major visible entities, attributes, actions, and relations; do not infer.
Return one compact JSON object with string keys enriched, paraphrase, saturated and no other text."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    natural = read_jsonl(args.staging / "natural_contexts.jsonl")
    contexts_by_image = {}
    for row in natural:
        contexts_by_image.setdefault(row["staging_image_id"], []).append(row)
    bases = [sorted(rows, key=lambda row: row["context_id"])[0]
             for _, rows in sorted(contexts_by_image.items())]
    for image_id, row in images.items():
        path = args.staging / row["image_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {image_id}")
    payload = {
        "inputs": {name: hashlib.sha256((args.staging / name).read_bytes()).hexdigest()
                   for name in ("staging_images.jsonl", "natural_contexts.jsonl")},
        "model": args.model, "revision": args.revision, "prompt": PROMPT,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "implementation": "e2-context-proposals-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "context_proposals.jsonl", args.output / "runtime.json"
    existing = {row["staging_image_id"]: row for row in read_jsonl(rows_path)}
    if metadata_path.is_file() and json.loads(metadata_path.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different proposal run")
    pending = [row for row in bases if row["staging_image_id"] not in existing]
    metadata = {
        "scope": "model-generated review proposals only; never E0/E2 gold",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "images": len(bases), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not pending:
        print("context proposal cache complete")
        return
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
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        chunk = pending[offset:offset + args.batch_size]
        texts, batch_images = [], []
        for context in chunk:
            prompt = PROMPT.format(caption=context["initial_caption"])
            conversation = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                       add_generation_prompt=True))
            image_row = images[context["staging_image_id"]]
            batch_images.append(Image.open(args.staging / image_row["image_path"]).convert("RGB"))
        inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt").to("cuda")
        batch_started = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        latency = round(1000 * (time.time() - batch_started) / len(chunk))
        for context, raw, output_ids in zip(chunk, decoded, trimmed):
            parsed = None
            try:
                parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            except (json.JSONDecodeError, AttributeError):
                pass
            valid = (isinstance(parsed, dict) and set(parsed) == {"enriched", "paraphrase", "saturated"}
                     and all(isinstance(value, str) and value.strip() for value in parsed.values()))
            existing[context["staging_image_id"]] = {
                "proposal_id": "proposal_" + hashlib.sha256(
                    context["staging_image_id"].encode()).hexdigest()[:20],
                "staging_image_id": context["staging_image_id"],
                "reference_context_id": context["context_id"],
                "reference_caption": context["initial_caption"],
                "proposals": parsed if valid else None, "raw_output": raw.strip(),
                "status": "parsed_review_required" if valid else "parse_failed_review_required",
                "output_tokens": int(output_ids.numel()), "latency_ms": latency,
            }
        for image in batch_images:
            image.close()
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"contexts {len(existing)}/{len(bases)}", flush=True)
    metadata.update({"status": "complete", "completed": len(existing),
                     "elapsed_seconds": time.time() - started,
                     "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     "gpu_name": torch.cuda.get_device_name(0)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
