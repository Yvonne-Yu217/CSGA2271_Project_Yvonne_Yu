"""Classify independently supported full-image completions by semantic target type."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl, read_jsonl
from acquisition.screen_core_target import PROMPT, STRICT_PROMPT, parse_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--full-image-screen-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-variant", choices=("default", "strict"), default="strict")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    judgments = {row["context_id"]: row for row in read_jsonl(
        args.full_image_screen_output / "judgments.jsonl")}
    if set(judgments) != set(contexts):
        raise RuntimeError("full-image judgments are incomplete")
    prompt_template = STRICT_PROMPT if args.prompt_variant == "strict" else PROMPT
    sources = [args.staging / "natural_contexts.jsonl",
               args.full_image_screen_output / "judgments.jsonl"]
    payload = {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "model": args.model, "revision": args.revision, "prompt": prompt_template,
        "batch_size": args.batch_size,
        "implementation": "full-image-completion-core-target-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "complement_types.jsonl", args.output / "runtime.json"
    existing = {row["context_id"]: row for row in read_jsonl(rows_path)}
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another completion semantic screen")
    pending = []
    for context_id in sorted(contexts):
        if context_id in existing:
            continue
        row = judgments[context_id]
        eligible = row["visual_label"] == "SUPPORTED" and "NEUTRAL" in row["nli_label"]
        if eligible:
            pending.append(contexts[context_id])
        else:
            existing[context_id] = {
                "context_id": context_id, "staging_image_id": row["staging_image_id"],
                "label": "NOT_NEW", "raw_output": "deterministic:not-eligible", "status": "ok",
            }
    metadata = {
        "scope": "automated semantic type screen of independently filtered full-image completions",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt_variant": args.prompt_variant,
        "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        "expected": len(contexts), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    if not pending and old:
        metadata = {**metadata, **old, "completed": len(existing), "status": "complete"}
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
            texts = []
            for context in chunk:
                observation = judgments[context["context_id"]]["completion"]
                prompt = prompt_template.format(caption=context["initial_caption"],
                                                observation=observation)
                conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=10,
                                           do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for context, raw in zip(chunk, decoded):
                label = parse_label(raw)
                existing[context["context_id"]] = {
                    "context_id": context["context_id"],
                    "staging_image_id": context["staging_image_id"],
                    "label": label, "raw_output": raw.strip(),
                    "status": "ok" if label else "parse_failed",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"completion core target {len(existing)}/{len(contexts)}", flush=True)
    failures = [row for row in existing.values() if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"unparsed completion semantic judgments: {len(failures)}")
    context_ids = sorted(contexts)
    values = [int(existing[key]["label"] == "MENTIONED_ENTITY_DETAIL") for key in context_ids]
    clusters = [contexts[key]["staging_image_id"] for key in context_ids]
    comparison_path = args.comparison_output / "provisional_report.json"
    comparison = json.loads(comparison_path.read_text())
    comparison_by_key = {(row["context_id"], row["method"]): row["core_target_success"]
                         for row in comparison["context_rows"]}
    methods = sorted({row["method"] for row in comparison["context_rows"]})
    gaps = {
        method: paired_bootstrap(
            values, [comparison_by_key[(key, method)] for key in context_ids], clusters,
            samples=10000, seed=2271)
        for method in methods
    }
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated semantic typing is not human fact adjudication.",
        "contexts": len(contexts), "images": len(set(clusters)),
        "type_counts": dict(Counter(row["label"] for row in existing.values())),
        "full_image_completion_core_target": bootstrap_cluster(values, clusters,
                                                               samples=10000, seed=2271),
        "full_image_completion_minus_comparison": gaps,
        "comparison_report_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        "comparison_methods": comparison["core_target_success"],
    }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    metadata.update({"status": "complete", "completed": len(existing),
                     "gpu_name": torch.cuda.get_device_name(0)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
