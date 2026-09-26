"""Generate one deterministic, content-preserving paraphrase per development caption."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from acquisition.observe import atomic_jsonl, read_jsonl


MODEL = "HuggingFaceTB/SmolVLM-Instruct"
REVISION = "81cd9a775a4d644f2faf4e7becff4559b46b14c7"
PROMPT = """Paraphrase the caption below as one natural sentence. Preserve every explicitly
stated entity, attribute, action, relation, and uncertainty. Do not add, remove, generalize, or
infer any visual fact. Reply with only the paraphrased sentence.
Caption: {caption}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    source = args.staging / "natural_contexts.jsonl"
    contexts = read_jsonl(source)
    payload = {
        "implementation": "content-preserving-caption-paraphrase-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "paraphrased_contexts.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different paraphrase run")
    existing = {row["context_id"]: row for row in read_jsonl(rows_path)}
    pending = [row for row in contexts if row["context_id"] not in existing]
    runtime = {**payload, "fingerprint": fingerprint, "expected": len(contexts),
               "completed": len(existing), "status": "running" if pending else "complete",
               "scope": "automated development paraphrases; semantic preservation is not human-adjudicated",
               "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    if pending:
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForVision2Seq.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            texts = [processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(
                    caption=row["initial_caption"])}]}], tokenize=False,
                add_generation_prompt=True) for row in chunk]
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=64,
                                           do_sample=False, use_cache=True)
            trimmed = [output[len(source_ids):]
                       for source_ids, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for row, text in zip(chunk, decoded):
                existing[row["context_id"]] = {
                    **row, "original_caption": row["initial_caption"],
                    "initial_caption": text.strip(),
                    "status": "ok" if text.strip() else "empty",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"paraphrases {len(existing)}/{len(contexts)}", flush=True)
    failures = sum(row.get("status") != "ok" for row in existing.values())
    runtime.update({"status": "complete" if not failures else "fail", "failures": failures,
                    "completed": len(existing), "gpu_name": torch.cuda.get_device_name(0)})
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"empty paraphrases: {failures}")


if __name__ == "__main__":
    main()
