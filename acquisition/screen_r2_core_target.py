"""Strict same-instance mentioned-entity sensitivity for R2 context actions."""
import argparse
import hashlib
import itertools
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import (DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl,
                                 read_jsonl)
from acquisition.screen_core_target import STRICT_PROMPT, parse_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path, required=True)
    parser.add_argument("--nli-output", type=Path, required=True)
    parser.add_argument("--full-image-core-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    observations = read_jsonl(args.observer_output / "observations.jsonl")
    visual = {row["cache_key"]: row for row in read_jsonl(
        args.visual_output / "visual_support.jsonl")}
    nli = {row["key"]: row for row in read_jsonl(args.nli_output / "entailments.jsonl")}
    if not (len(observations) == len(visual) == len(nli) == 7000):
        raise RuntimeError("R2 inputs must each contain 7000 rows")
    work = []
    for row in observations:
        key = row["cache_key"]
        work.append({
            "key": key, "context_id": row["context_id"],
            "staging_image_id": row["staging_image_id"],
            "candidate_id": row["candidate_id"],
            "caption": contexts[row["context_id"]]["initial_caption"],
            "observation": row["observed_text"],
            "eligible": visual[key]["label"] == "SUPPORTED" and "NEUTRAL" in nli[key]["label"],
        })
    source_paths = [args.staging / "natural_contexts.jsonl",
                    args.observer_output / "observations.jsonl",
                    args.visual_output / "visual_support.jsonl",
                    args.nli_output / "entailments.jsonl"]
    payload = {
        "implementation": "r2-strict-core-target-caption-necessity-v1",
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths},
        "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(STRICT_PROMPT.encode()).hexdigest(),
        "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "complement_types.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different R2 core-target fingerprint")
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    for row in work:
        if row["key"] not in existing and not row["eligible"]:
            existing[row["key"]] = {
                **{key: row[key] for key in ("key", "context_id", "staging_image_id",
                                             "candidate_id")},
                "label": "NOT_NEW", "raw_output": "deterministic:not-eligible",
                "status": "ok",
            }
    pending = [row for row in work if row["key"] not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "automated strict semantic typing; not human gold",
        "expected": len(work), "eligible": len(pending), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, local_files_only=True, use_fast=False)
        processor.tokenizer.padding_side = "left"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            texts = []
            for row in chunk:
                prompt = STRICT_PROMPT.format(caption=row["caption"],
                                              observation=row["observation"])
                conversation = [{"role": "user", "content": [
                    {"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=10,
                                           do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for row, raw in zip(chunk, decoded):
                label = parse_label(raw)
                existing[row["key"]] = {
                    **{key: row[key] for key in ("key", "context_id", "staging_image_id",
                                                 "candidate_id")},
                    "label": label, "raw_output": raw.strip(),
                    "status": "ok" if label else "parse_failed",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({
                "completed": len(existing), "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"R2 strict target {len(existing)}/{len(work)}", flush=True)
    failures = [row for row in existing.values() if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"unparsed R2 strict target outputs: {len(failures)}")
    success = {(row["context_id"], row["candidate_id"]):
               int(existing[row["key"]]["label"] == "MENTIONED_ENTITY_DETAIL")
               for row in work}
    contexts_by_image, candidates_by_image = defaultdict(set), defaultdict(set)
    for row in work:
        contexts_by_image[row["staging_image_id"]].add(row["context_id"])
        candidates_by_image[row["staging_image_id"]].add(row["candidate_id"])
    oracle, static, clusters, set_change, pair_jaccard = [], [], [], [], []
    ordered_context_ids = []
    for image_id in sorted(contexts_by_image):
        context_ids = sorted(contexts_by_image[image_id])
        candidate_ids = sorted(candidates_by_image[image_id])
        best_static = max(candidate_ids, key=lambda candidate_id: (
            sum(success[(context_id, candidate_id)] for context_id in context_ids), candidate_id))
        sets = []
        for context_id in context_ids:
            values = {candidate_id for candidate_id in candidate_ids
                      if success[(context_id, candidate_id)]}
            sets.append(values)
            ordered_context_ids.append(context_id)
            oracle.append(int(bool(values)))
            static.append(success[(context_id, best_static)])
            clusters.append(image_id)
        for left, right in itertools.combinations(sets, 2):
            union = left | right
            pair_jaccard.append(len(left & right) / len(union) if union else 1.0)
            set_change.append(int(left != right))
    full_image = {row["context_id"]: row for row in read_jsonl(
        args.full_image_core_output / "complement_types.jsonl")}
    direct = [int(full_image[context_id]["label"] == "MENTIONED_ENTITY_DETAIL")
              for context_id in ordered_context_ids]
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated semantic typing is not human fact adjudication.",
        "contexts": len(oracle), "images": len(contexts_by_image),
        "type_counts": dict(Counter(row["label"] for row in existing.values())),
        "strict_residual_oracle": bootstrap_cluster(oracle, clusters, samples=10000, seed=2271),
        "best_same_image_static": bootstrap_cluster(static, clusters, samples=10000, seed=2271),
        "caption_specific_oracle_minus_static": paired_bootstrap(
            oracle, static, clusters, samples=10000, seed=2271),
        "full_image_completion": bootstrap_cluster(direct, clusters, samples=10000, seed=2271),
        "strict_oracle_minus_full_image": paired_bootstrap(
            oracle, direct, clusters, samples=10000, seed=2271),
        "same_image_caption_pair_action_set_change_rate": sum(set_change) / len(set_change),
        "same_image_caption_pair_action_set_jaccard": sum(pair_jaccard) / len(pair_jaccard),
    }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({
        "status": "complete", "completed": len(existing),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
