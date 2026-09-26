"""Caption-conditioned full-image completion baseline with resumable caching."""
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


PROMPT = """Existing image description: {caption}
Inspect the full image and add the single most useful concrete visible fact not already stated or
entailed by the description. Prefer an attribute, action, state, or relation of a specific entity
already mentioned when one is clearly visible. Do not infer identity, intent, location, or unseen
context. Reply with one sentence of at most 30 words and no heading. If no useful new visible fact
can be added, reply exactly NO_NEW_FACT."""

REFINEMENT_PROMPT = """Existing image description: {caption}
First full-image completion: {prior_completion}
Inspect the full image again and add the single most useful concrete visible fact not already
stated or entailed by either text above. Prefer an attribute, action, state, or relation of a
specific mentioned entity. Do not repeat or paraphrase the first completion. Do not infer identity,
intent, location, or unseen context. Reply with one sentence of at most 30 words and no heading.
If no distinct useful new visible fact can be added, reply exactly NO_NEW_FACT."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-pixels", type=int, default=0,
                        help="Optional Qwen image pixel cap; zero uses the model default")
    parser.add_argument("--prior-completion-output", type=Path,
                        help="Completed first-pass output; enables equal-call refinement")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.max_pixels < 0:
        parser.error("max-pixels must be zero or positive")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = read_jsonl(args.staging / "natural_contexts.jsonl")
    sources = [args.staging / "staging_images.jsonl", args.staging / "natural_contexts.jsonl"]
    priors = {}
    if args.prior_completion_output:
        prior_path = args.prior_completion_output / "completions.jsonl"
        sources.append(prior_path)
        prior_rows = read_jsonl(prior_path)
        priors = {row["context_id"]: row["completion"] for row in prior_rows}
        if (set(priors) != {row["context_id"] for row in contexts} or
                any(row.get("status") != "ok" for row in prior_rows)):
            raise RuntimeError("first-pass completion cache is incomplete")
    prompt_template = REFINEMENT_PROMPT if priors else PROMPT
    for image_id, row in images.items():
        image_path = args.staging / row["image_path"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {image_id}")
    payload = {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "model": args.model, "revision": args.revision, "prompt": prompt_template,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "implementation": ("caption-conditioned-full-image-refinement-v1" if priors else
                           "caption-conditioned-full-image-completion-v1"),
    }
    if args.max_pixels:
        payload["max_pixels"] = args.max_pixels
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "completions.jsonl", args.output / "runtime.json"
    existing = {row["context_id"]: row for row in read_jsonl(rows_path)}
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another full-image completion run")
    pending = [row for row in contexts if row["context_id"] not in existing]
    metadata = {
        "scope": "model-visible full-image completion baseline; automated output is not gold",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        "contexts": len(contexts), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    if not pending and old:
        metadata = {**metadata, **old, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if not pending:
        print("full-image completion cache complete")
        return
    processor_kwargs = {"revision": args.revision,
                        "local_files_only": not args.allow_model_download,
                        "use_fast": False}
    if args.max_pixels:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
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
        batch_images, texts = [], []
        for context in chunk:
            image_row = images[context["staging_image_id"]]
            batch_images.append(Image.open(args.staging / image_row["image_path"]).convert("RGB"))
            conversation = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template.format(
                    caption=context["initial_caption"],
                    prior_completion=priors.get(context["context_id"], ""))}]}]
            texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                       add_generation_prompt=True))
        inputs = processor(text=texts, images=batch_images, padding=True,
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        for context, raw in zip(chunk, decoded):
            text = raw.strip()
            existing[context["context_id"]] = {
                "context_id": context["context_id"],
                "staging_image_id": context["staging_image_id"],
                "completion": text,
                "status": "ok" if text else "empty",
            }
        for image in batch_images:
            image.close()
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"full-image completion {len(existing)}/{len(contexts)}", flush=True)
    failures = [row for row in existing.values() if row["status"] != "ok"]
    metadata.update({"status": "complete" if not failures else "fail",
                     "completed": len(existing), "elapsed_seconds": time.time() - started,
                     "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                     "gpu_name": torch.cuda.get_device_name(0), "failures": len(failures)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"empty full-image completions: {len(failures)}")


if __name__ == "__main__":
    main()
