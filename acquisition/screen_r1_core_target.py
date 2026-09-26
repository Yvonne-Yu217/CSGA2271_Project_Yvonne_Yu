"""Strict mentioned-entity sensitivity for independently filtered R1 outputs."""
import argparse
import hashlib
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


IMPLEMENTATION = "r1-strict-core-target-v1"


def parse_arm(value):
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("arm must be NAME=OBSERVER_DIR=SUPPORT_DIR")
    return parts[0], Path(parts[1]), Path(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--nli-output", type=Path, required=True)
    parser.add_argument("--full-image-core-output", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    nli_rows = {row["key"]: row
                for row in read_jsonl(args.nli_output / "entailments.jsonl")}
    sources = [args.staging / "natural_contexts.jsonl",
               args.nli_output / "entailments.jsonl"]
    work = []
    for arm, observer_dir, support_dir in args.arm:
        observation_path = observer_dir / "observations.jsonl"
        support_path = support_dir / "visual_support.jsonl"
        sources.extend([observation_path, support_path])
        supports = {row["candidate_id"]: row["label"]
                    for row in read_jsonl(support_path)}
        for observation in read_jsonl(observation_path):
            key = f"{arm}:{observation['context_id']}:{observation['candidate_id']}"
            nli = nli_rows.get(key)
            if nli is None:
                raise RuntimeError(f"missing NLI row: {key}")
            work.append({
                "key": key, "arm": arm, "context_id": observation["context_id"],
                "staging_image_id": observation["staging_image_id"],
                "candidate_id": observation["candidate_id"],
                "caption": contexts[observation["context_id"]]["initial_caption"],
                "observation": observation["observed_text"],
                "eligible": supports[observation["candidate_id"]] == "SUPPORTED"
                            and "NEUTRAL" in nli["label"],
            })
    payload = {
        "implementation": IMPLEMENTATION,
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sources},
        "model": args.model, "revision": args.revision,
        "prompt_sha256": hashlib.sha256(STRICT_PROMPT.encode()).hexdigest(),
        "batch_size": args.batch_size, "arms": [name for name, _, _ in args.arm],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "complement_types.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different R1 core-target fingerprint")
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    for row in work:
        if row["key"] not in existing and not row["eligible"]:
            existing[row["key"]] = {
                **{key: row[key] for key in ("key", "arm", "context_id",
                                             "staging_image_id", "candidate_id")},
                "label": "NOT_NEW", "raw_output": "deterministic:not-eligible",
                "status": "ok",
            }
    pending = [row for row in work if row["key"] not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "automated strict semantic typing; not human gold",
        "expected": len(work), "eligible": len(pending), "completed": len(existing),
        "status": "running" if pending else "complete",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
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
                    **{key: row[key] for key in ("key", "arm", "context_id",
                                                 "staging_image_id", "candidate_id")},
                    "label": label, "raw_output": raw.strip(),
                    "status": "ok" if label else "parse_failed",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({
                "completed": len(existing), "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"R1 strict target {len(existing)}/{len(work)}", flush=True)
    ordered = [existing[row["key"]] for row in work]
    failures = [row for row in ordered if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"unparsed R1 strict target outputs: {len(failures)}")
    by_context = defaultdict(list)
    for row in ordered:
        by_context[(row["arm"], row["context_id"])].append(
            int(row["label"] == "MENTIONED_ENTITY_DETAIL"))
    full_image_rows = {row["context_id"]: row for row in read_jsonl(
        args.full_image_core_output / "complement_types.jsonl")}
    source_full_path = args.full_image_core_output / "complement_types.jsonl"
    summaries, vectors = {}, {}
    for arm, _, _ in args.arm:
        arm_rows = [row for row in ordered if row["arm"] == arm]
        context_ids = sorted({row["context_id"] for row in arm_rows})
        values = [int(any(by_context[(arm, context_id)])) for context_id in context_ids]
        baseline = [int(full_image_rows[context_id]["label"] == "MENTIONED_ENTITY_DETAIL")
                    for context_id in context_ids]
        clusters = [contexts[context_id]["staging_image_id"] for context_id in context_ids]
        vectors[arm] = (context_ids, values, clusters)
        summaries[arm] = {
            "type_counts": dict(Counter(row["label"] for row in arm_rows)),
            "strict_residual_oracle": bootstrap_cluster(
                values, clusters, samples=10000, seed=2271),
            "full_image_completion": bootstrap_cluster(
                baseline, clusters, samples=10000, seed=2271),
            "oracle_minus_full_image": paired_bootstrap(
                values, baseline, clusters, samples=10000, seed=2271),
        }
    names = [name for name, _, _ in args.arm]
    comparisons = {}
    for left, right in ((names[1], names[0]), (names[3], names[2]),
                        (names[0], names[2]), (names[1], names[3])):
        left_ids, left_values, clusters = vectors[left]
        right_ids, right_values, _ = vectors[right]
        if left_ids != right_ids:
            raise RuntimeError("R1 arms use different contexts")
        comparisons[f"{left}_minus_{right}"] = paired_bootstrap(
            left_values, right_values, clusters, samples=10000, seed=2271)
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated semantic typing is not human fact adjudication.",
        "full_image_core_sha256": hashlib.sha256(source_full_path.read_bytes()).hexdigest(),
        "arms": summaries, "paired_strict_oracle_differences": comparisons,
    }
    (args.output / "provisional_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    runtime.update({
        "status": "complete", "completed": len(ordered),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
