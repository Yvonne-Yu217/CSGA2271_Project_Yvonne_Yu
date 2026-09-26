"""Independent novelty and caption-necessity diagnostics for R2."""
import argparse
import hashlib
import itertools
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from acquisition.metrics import bootstrap_cluster, paired_bootstrap
from acquisition.observe import atomic_jsonl, read_jsonl
from acquisition.screen_independent_nli import DEFAULT_MODEL, DEFAULT_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path, required=True)
    parser.add_argument("--full-image-screen-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=0,
                        help="Optional exact row count")
    parser.add_argument("--batch-size", type=int, default=256)
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
    if (len(observations) != len(visual) or
            (args.expected and len(observations) != args.expected)):
        raise RuntimeError("R2 observations/visual judgments have unexpected row counts")
    work = []
    for row in observations:
        if row["cache_key"] not in visual or visual[row["cache_key"]].get("status") != "ok":
            raise RuntimeError(f"missing visual judgment: {row['cache_key']}")
        work.append({
            "key": row["cache_key"], "context_id": row["context_id"],
            "staging_image_id": row["staging_image_id"],
            "candidate_id": row["candidate_id"],
            "premise": contexts[row["context_id"]]["initial_caption"],
            "hypothesis": row["observed_text"],
        })
    source_paths = [args.staging / "natural_contexts.jsonl",
                    args.observer_output / "observations.jsonl",
                    args.visual_output / "visual_support.jsonl"]
    payload = {
        "implementation": "r2-independent-nli-caption-necessity-v1",
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths},
        "model": args.model, "revision": args.revision, "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "entailments.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different R2 NLI fingerprint")
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    for row in work:
        if row["key"] not in existing and row["hypothesis"].strip().upper() in {
                "NO_NEW_FACT", "NO_VISIBLE_FACT"}:
            existing[row["key"]] = {
                **{key: row[key] for key in ("key", "context_id", "staging_image_id",
                                             "candidate_id")},
                "label": "UNINFORMATIVE", "probabilities": {}, "status": "ok",
            }
    pending = [row for row in work if row["key"] not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "automated independent NLI sensitivity; not human gold",
        "expected": len(work), "completed": len(existing),
        "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision, local_files_only=True,
            dtype=torch.float16).to("cuda").eval()
        id2label = {int(key): value.upper() for key, value in model.config.id2label.items()}
        runtime["id2label"] = id2label
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            inputs = tokenizer(
                [row["premise"] for row in chunk], [row["hypothesis"] for row in chunk],
                padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits.float().softmax(-1).cpu()
            for row, distribution in zip(chunk, probabilities):
                index = int(distribution.argmax())
                existing[row["key"]] = {
                    **{key: row[key] for key in ("key", "context_id", "staging_image_id",
                                                 "candidate_id")},
                    "label": id2label[index],
                    "probabilities": {id2label[i]: float(value)
                                      for i, value in enumerate(distribution)}, "status": "ok",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({
                "completed": len(existing), "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"R2 independent NLI {len(existing)}/{len(work)}", flush=True)
    labels = {row["label"] for row in existing.values()} - {"UNINFORMATIVE"}
    neutral_labels = {label for label in labels if "NEUTRAL" in label}
    if len(neutral_labels) != 1:
        raise RuntimeError(f"could not identify neutral label: {sorted(labels)}")
    neutral = next(iter(neutral_labels))
    success = {}
    contexts_by_image, candidates_by_image = defaultdict(set), defaultdict(set)
    for row in work:
        success[(row["context_id"], row["candidate_id"])] = int(
            visual[row["key"]]["label"] == "SUPPORTED" and
            existing[row["key"]]["label"] == neutral)
        contexts_by_image[row["staging_image_id"]].add(row["context_id"])
        candidates_by_image[row["staging_image_id"]].add(row["candidate_id"])
    oracle, static, clusters = [], [], []
    set_change, pair_jaccard = [], []
    for image_id in sorted(contexts_by_image):
        context_ids = sorted(contexts_by_image[image_id])
        candidate_ids = sorted(candidates_by_image[image_id])
        best_static = max(candidate_ids, key=lambda candidate_id: (
            sum(success[(context_id, candidate_id)] for context_id in context_ids),
            candidate_id))
        sets = []
        for context_id in context_ids:
            values = {candidate_id for candidate_id in candidate_ids
                      if success[(context_id, candidate_id)]}
            sets.append(values)
            oracle.append(int(bool(values)))
            static.append(success[(context_id, best_static)])
            clusters.append(image_id)
        for left, right in itertools.combinations(sets, 2):
            union = left | right
            pair_jaccard.append(len(left & right) / len(union) if union else 1.0)
            set_change.append(int(left != right))
    direct_rows = {row["context_id"]: row for row in read_jsonl(
        args.full_image_screen_output / "judgments.jsonl")}
    ordered_context_ids = [context_id for image_id in sorted(contexts_by_image)
                           for context_id in sorted(contexts_by_image[image_id])]
    direct = [int(direct_rows[context_id]["visual_label"] == "SUPPORTED" and
                  "NEUTRAL" in direct_rows[context_id]["nli_label"])
              for context_id in ordered_context_ids]
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated SmolVLM support plus DeBERTa novelty; not human adjudication.",
        "contexts": len(oracle), "images": len(contexts_by_image),
        "nli_counts": dict(Counter(row["label"] for row in existing.values())),
        "residual_oracle": bootstrap_cluster(oracle, clusters, samples=10000, seed=2271),
        "best_same_image_static": bootstrap_cluster(static, clusters, samples=10000, seed=2271),
        "caption_specific_oracle_minus_static": paired_bootstrap(
            oracle, static, clusters, samples=10000, seed=2271),
        "full_image_completion": bootstrap_cluster(direct, clusters, samples=10000, seed=2271),
        "residual_oracle_minus_full_image": paired_bootstrap(
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
