"""Measure whether fixed high-resolution crops can verify full-image generated claims."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from acquisition.metrics import bootstrap_cluster
from acquisition.observe import atomic_jsonl, crop_candidate, read_jsonl
from acquisition.screen_independent_visual import (DEFAULT_MODEL, DEFAULT_REVISION,
                                                   parse_independent_label)
from acquisition.screen_visual_support import PROMPT


RETRY_PROMPT = (PROMPT + "\nDo not transcribe image text or describe the crop. "
                "Output one label from the four-label list only.")
RETRY_CODE_PROMPT = """Claim: {observation}
Judge the claim only against the supplied crop. Choose exactly one code:
A = every concrete claim is directly visible
B = some but not every concrete claim is visible
C = no concrete claim is supported or the claim relies on inference
D = the claim states no recognizable concrete fact
Reply with only A, B, C, or D. Do not transcribe any image text."""
CODE_LABELS = {"A": "SUPPORTED", "B": "PARTIAL", "C": "INVALID", "D": "UNINFORMATIVE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--full-image-screen-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    candidates_by_image = defaultdict(list)
    candidate_lookup = {}
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        if row["kind"] != "stop":
            candidates_by_image[row["staging_image_id"]].append(row)
            candidate_lookup[row["candidate_id"]] = row
    claims = {row["context_id"]: row
              for row in read_jsonl(args.full_image_screen_output / "judgments.jsonl")}
    if set(claims) != set(contexts):
        raise RuntimeError("full-image claim judgments are incomplete")
    work = []
    for context_id in sorted(contexts):
        context = contexts[context_id]
        for candidate in sorted(candidates_by_image[context["staging_image_id"]],
                                key=lambda row: row["candidate_id"]):
            work.append({"key": context_id + ":" + candidate["candidate_id"],
                         "context_id": context_id,
                         "staging_image_id": context["staging_image_id"],
                         "candidate_id": candidate["candidate_id"],
                         "claim": claims[context_id]["completion"],
                         "candidate": candidate})
    sources = [args.staging / "staging_images.jsonl",
               args.staging / "natural_contexts.jsonl",
               args.staging / "automatic_candidates.jsonl",
               args.full_image_screen_output / "judgments.jsonl"]
    payload = {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "model": args.model, "revision": args.revision, "prompt": PROMPT,
        "batch_size": args.batch_size,
        "implementation": "claim-conditioned-crop-support-v1-left-pad-768px",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "crop_support.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another claim-crop screen")
    pending = [row for row in work if existing.get(row["key"], {}).get("status") != "ok"]
    metadata = {
        "scope": "automated crop-verification oracle screen; not human gold",
        "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "contexts": len(contexts), "expected": len(work), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "retry_prompt_sha256": hashlib.sha256(RETRY_PROMPT.encode()).hexdigest(),
        "retry_code_prompt_sha256": hashlib.sha256(RETRY_CODE_PROMPT.encode()).hexdigest(),
    }
    if not pending and old:
        metadata = {**metadata, **old, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, local_files_only=not args.allow_model_download,
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
        prior_elapsed = float(old.get("elapsed_seconds", 0)) if old else 0.0
        prior_peak = float(old.get("peak_gpu_memory_mb", 0)) if old else 0.0
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            crops, texts, retry_flags = [], [], []
            for row in chunk:
                crops.append(crop_candidate(args.staging, images[row["staging_image_id"]],
                                            row["candidate"]))
                previous = existing.get(row["key"], {})
                if previous.get("status") != "parse_failed":
                    prompt_variant, prompt = "base", PROMPT
                elif previous.get("prompt_variant") == "retry_format":
                    prompt_variant, prompt = "retry_code", RETRY_CODE_PROMPT
                else:
                    prompt_variant, prompt = "retry_format", RETRY_PROMPT
                retry_flags.append(prompt_variant)
                conversation = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt.format(observation=row["claim"])}]}]
                texts.append(processor.apply_chat_template(conversation,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, images=crops, padding=True,
                               return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16,
                                           do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            for row, raw, prompt_variant in zip(chunk, decoded, retry_flags):
                label = parse_independent_label(raw)
                if not label and prompt_variant == "retry_code":
                    label = CODE_LABELS.get(raw.strip().upper().strip(".:-"))
                existing[row["key"]] = {
                    "key": row["key"], "context_id": row["context_id"],
                    "staging_image_id": row["staging_image_id"],
                    "candidate_id": row["candidate_id"], "label": label,
                    "raw_output": raw.strip(), "status": "ok" if label else "parse_failed",
                    "prompt_variant": prompt_variant,
                }
            for crop in crops:
                crop.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({
                "completed": len(existing),
                "elapsed_seconds": prior_elapsed + time.time() - started,
                "peak_gpu_memory_mb": max(prior_peak,
                    torch.cuda.max_memory_allocated() / 1024 ** 2)})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"claim crop support {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": prior_elapsed + time.time() - started,
                         "peak_gpu_memory_mb": max(prior_peak,
                             torch.cuda.max_memory_allocated() / 1024 ** 2),
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    failures = [row for row in existing.values() if row.get("status") != "ok"]
    if len(existing) != len(work) or failures:
        raise RuntimeError(f"claim-crop support cache incomplete/unparsed: {len(failures)}")
    by_context = defaultdict(list)
    for row in existing.values():
        by_context[row["context_id"]].append(row)
    summary_rows = []
    for context_id in sorted(contexts):
        crop_rows = by_context[context_id]
        image_row = images[contexts[context_id]["staging_image_id"]]
        full_area = image_row["width"] * image_row["height"]
        global_supported = claims[context_id]["visual_label"] == "SUPPORTED"
        any_crop_supported = any(row["label"] == "SUPPORTED" for row in crop_rows)
        zoom_rows = [row for row in crop_rows
                     if candidate_lookup[row["candidate_id"]]["pixel_cost"] < full_area]
        local_rows = [row for row in crop_rows
                      if candidate_lookup[row["candidate_id"]]["pixel_cost"] <= full_area / 4]
        summary_rows.append({
            "context_id": context_id,
            "staging_image_id": contexts[context_id]["staging_image_id"],
            "global_visual_label": claims[context_id]["visual_label"],
            "global_supported": int(global_supported),
            "any_crop_supported": int(any_crop_supported),
            "any_zoom_supported": int(any(row["label"] == "SUPPORTED" for row in zoom_rows)),
            "any_local_supported": int(any(row["label"] == "SUPPORTED" for row in local_rows)),
            "agreement": int(global_supported == any_crop_supported),
            "supported_crop_count": sum(row["label"] == "SUPPORTED" for row in crop_rows),
        })
    clusters = [row["staging_image_id"] for row in summary_rows]
    supported = [row for row in summary_rows if row["global_supported"]]
    unsupported = [row for row in summary_rows if not row["global_supported"]]
    report = {
        "status": "provisional_not_evidence",
        "warning": ("Full-image reference and crop judgments use the same SmolVLM model; "
                    "this tests candidate coverage, not human truth."),
        "contexts": len(summary_rows), "images": len(set(clusters)),
        "crop_label_counts": dict(Counter(row["label"] for row in existing.values())),
        "global_label_counts": dict(Counter(row["global_visual_label"] for row in summary_rows)),
        "any_crop_global_agreement": bootstrap_cluster(
            [row["agreement"] for row in summary_rows], clusters, samples=10000, seed=2271),
        "supported_claim_crop_recall": bootstrap_cluster(
            [row["any_crop_supported"] for row in supported],
            [row["staging_image_id"] for row in supported], samples=10000, seed=2271),
        "unsupported_claim_false_support": bootstrap_cluster(
            [row["any_crop_supported"] for row in unsupported],
            [row["staging_image_id"] for row in unsupported], samples=10000, seed=2271),
        "zoom_supported_claim_recall": bootstrap_cluster(
            [row["any_zoom_supported"] for row in supported],
            [row["staging_image_id"] for row in supported], samples=10000, seed=2271),
        "zoom_unsupported_claim_false_support": bootstrap_cluster(
            [row["any_zoom_supported"] for row in unsupported],
            [row["staging_image_id"] for row in unsupported], samples=10000, seed=2271),
        "local_supported_claim_recall": bootstrap_cluster(
            [row["any_local_supported"] for row in supported],
            [row["staging_image_id"] for row in supported], samples=10000, seed=2271),
        "local_unsupported_claim_false_support": bootstrap_cluster(
            [row["any_local_supported"] for row in unsupported],
            [row["staging_image_id"] for row in unsupported], samples=10000, seed=2271),
        "candidate_sets": {"all_including_full_image": 14,
                           "proper_zoom_excluding_full_image": 13,
                           "local_quarter_area_or_less": 9},
        "mean_supported_crops": sum(row["supported_crop_count"] for row in summary_rows) /
                                len(summary_rows),
        "context_rows": summary_rows,
    }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "context_rows"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
