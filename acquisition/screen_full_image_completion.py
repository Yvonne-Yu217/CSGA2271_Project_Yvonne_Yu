"""Independently screen caption-conditioned full-image completion outputs."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import (AutoModelForSequenceClassification, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer)

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import atomic_jsonl, read_jsonl
from acquisition.screen_independent_nli import (DEFAULT_MODEL as NLI_MODEL,
                                                DEFAULT_REVISION as NLI_REVISION)
from acquisition.screen_independent_visual import (DEFAULT_MODEL as VISUAL_MODEL,
                                                   DEFAULT_REVISION as VISUAL_REVISION,
                                                   parse_independent_label)
from acquisition.screen_visual_support import PROMPT as VISUAL_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-model", default=VISUAL_MODEL)
    parser.add_argument("--visual-revision", default=VISUAL_REVISION)
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--nli-revision", default=NLI_REVISION)
    parser.add_argument("--visual-batch-size", type=int, default=4)
    parser.add_argument("--nli-batch-size", type=int, default=64)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    completions = {row["context_id"]: row
                   for row in read_jsonl(args.completion_output / "completions.jsonl")}
    if set(completions) != set(contexts) or any(row.get("status") != "ok"
                                                 for row in completions.values()):
        raise RuntimeError("full-image completion cache is incomplete")
    sources = [args.staging / "staging_images.jsonl",
               args.staging / "natural_contexts.jsonl",
               args.completion_output / "completions.jsonl"]
    payload = {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "visual_model": args.visual_model, "visual_revision": args.visual_revision,
        "nli_model": args.nli_model, "nli_revision": args.nli_revision,
        "visual_prompt": VISUAL_PROMPT, "visual_batch_size": args.visual_batch_size,
        "nli_batch_size": args.nli_batch_size,
        "implementation": "independent-full-image-completion-screen-v1",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "judgments.jsonl", args.output / "runtime.json"
    existing = {row["context_id"]: row for row in read_jsonl(rows_path)}
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another full-image screen")
    for context_id, completion in completions.items():
        if context_id not in existing:
            existing[context_id] = {
                "context_id": context_id,
                "staging_image_id": completion["staging_image_id"],
                "completion": completion["completion"],
            }
        if completion["completion"].strip().upper() == "NO_NEW_FACT":
            existing[context_id].update({
                "visual_label": "UNINFORMATIVE", "visual_raw": "deterministic:NO_NEW_FACT",
                "visual_status": "ok", "nli_label": "UNINFORMATIVE",
                "nli_probabilities": {}, "nli_status": "ok",
            })
    visual_pending = [contexts[key] for key in sorted(contexts)
                      if existing[key].get("visual_status") != "ok"]
    metadata = {
        "scope": "independent automated screen of full-image completion; not human gold",
        "fingerprint": fingerprint, "visual_model": args.visual_model,
        "visual_revision": args.visual_revision, "nli_model": args.nli_model,
        "nli_revision": args.nli_revision, "expected": len(contexts),
        "visual_completed": len(contexts) - len(visual_pending), "status": "running",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if visual_pending:
        processor = AutoProcessor.from_pretrained(
            args.visual_model, revision=args.visual_revision,
            local_files_only=not args.allow_model_download, size={"longest_edge": 768})
        processor.tokenizer.padding_side = "left"
        model = AutoModelForVision2Seq.from_pretrained(
            args.visual_model, revision=args.visual_revision,
            local_files_only=not args.allow_model_download, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(visual_pending), args.visual_batch_size):
            chunk = visual_pending[offset:offset + args.visual_batch_size]
            batch_images, texts = [], []
            for context in chunk:
                image_row = images[context["staging_image_id"]]
                batch_images.append(Image.open(args.staging / image_row["image_path"]).convert("RGB"))
                observation = existing[context["context_id"]]["completion"]
                conversation = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": VISUAL_PROMPT.format(observation=observation)}]}]
                texts.append(processor.apply_chat_template(conversation,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, images=batch_images, padding=True,
                               return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16,
                                           do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            for context, raw in zip(chunk, decoded):
                label = parse_independent_label(raw)
                existing[context["context_id"]].update({
                    "visual_label": label, "visual_raw": raw.strip(),
                    "visual_status": "ok" if label else "parse_failed",
                })
            for image in batch_images:
                image.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({
                "visual_completed": sum(row.get("visual_status") == "ok"
                                        for row in existing.values()),
                "visual_elapsed_seconds": time.time() - started,
                "visual_peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"full-image visual screen {metadata['visual_completed']}/{len(contexts)}",
                  flush=True)
        del model, processor
        torch.cuda.empty_cache()
    visual_failures = [row for row in existing.values() if row.get("visual_status") != "ok"]
    if visual_failures:
        raise RuntimeError(f"unparsed full-image visual judgments: {len(visual_failures)}")
    nli_pending = [contexts[key] for key in sorted(contexts)
                   if existing[key].get("nli_status") != "ok"]
    if nli_pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.nli_model, revision=args.nli_revision,
            local_files_only=not args.allow_model_download)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.nli_model, revision=args.nli_revision,
            local_files_only=not args.allow_model_download, dtype=torch.float16).to("cuda").eval()
        id2label = {int(key): value.upper() for key, value in model.config.id2label.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(nli_pending), args.nli_batch_size):
            chunk = nli_pending[offset:offset + args.nli_batch_size]
            inputs = tokenizer(
                [row["initial_caption"] for row in chunk],
                [existing[row["context_id"]]["completion"] for row in chunk],
                padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits.float().softmax(-1).cpu()
            for context, distribution in zip(chunk, probabilities):
                label_index = int(distribution.argmax())
                existing[context["context_id"]].update({
                    "nli_label": id2label[label_index],
                    "nli_probabilities": {id2label[i]: float(value)
                                          for i, value in enumerate(distribution)},
                    "nli_status": "ok",
                })
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({
                "nli_completed": sum(row.get("nli_status") == "ok"
                                     for row in existing.values()),
                "nli_elapsed_seconds": time.time() - started,
                "nli_peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"full-image NLI screen {metadata['nli_completed']}/{len(contexts)}", flush=True)
    labels = {row["nli_label"] for row in existing.values()} - {"UNINFORMATIVE"}
    neutral_labels = {label for label in labels if "NEUTRAL" in label}
    if len(neutral_labels) != 1:
        raise RuntimeError(f"could not identify unique neutral label: {sorted(labels)}")
    neutral = next(iter(neutral_labels))
    ordered = [existing[key] for key in sorted(existing)]
    values = [int(row["visual_label"] == "SUPPORTED" and row["nli_label"] == neutral)
              for row in ordered]
    clusters = [row["staging_image_id"] for row in ordered]
    comparison_path = args.comparison_output / "provisional_report.json"
    comparison = json.loads(comparison_path.read_text())
    comparison_hash = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    direct_by_context = {row["context_id"]: value for row, value in zip(ordered, values)}
    comparison_by_key = {(row["context_id"], row["method"]): row["new_correct"]
                         for row in comparison["context_rows"]}
    compared_methods = sorted({row["method"] for row in comparison["context_rows"]})
    context_ids = sorted(contexts)
    paired_gaps = {
        method: paired_bootstrap(
            [direct_by_context[context_id] for context_id in context_ids],
            [comparison_by_key[(context_id, method)] for context_id in context_ids],
            [contexts[context_id]["staging_image_id"] for context_id in context_ids],
            samples=10000, seed=2271)
        for method in compared_methods
    }
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated independent-model sensitivity only; no human fact adjudication.",
        "contexts": len(ordered), "images": len(set(clusters)),
        "completion_no_new_fact": sum(row["completion"].strip().upper() == "NO_NEW_FACT"
                                      for row in ordered),
        "visual_counts": dict(Counter(row["visual_label"] for row in ordered)),
        "nli_counts": dict(Counter(row["nli_label"] for row in ordered)),
        "full_image_completion_success": bootstrap_cluster(values, clusters, samples=10000,
                                                            seed=2271),
        "comparison_report_sha256": comparison_hash,
        "comparison_methods": comparison["methods"],
        "full_image_completion_minus_comparison": paired_gaps,
    }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    metadata.update({"status": "complete", "visual_completed": len(ordered),
                     "nli_completed": len(ordered), "gpu_name": torch.cuda.get_device_name(0)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
