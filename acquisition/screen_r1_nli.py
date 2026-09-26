"""Independent NLI sensitivity and residual-oracle summary for R1 arms."""
import argparse
import hashlib
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


IMPLEMENTATION = "r1-independent-nli-v1"


def parse_arm(value):
    parts = value.split("=", 2)
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("arm must be NAME=OBSERVER_DIR=SUPPORT_DIR")
    return parts[0], Path(parts[1]), Path(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if len({name for name, _, _ in args.arm}) != len(args.arm):
        parser.error("arm names must be unique")
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    source_paths = [args.staging / "natural_contexts.jsonl"]
    work, visual_by_key = [], {}
    for arm, observer_dir, support_dir in args.arm:
        observation_path = observer_dir / "observations.jsonl"
        support_path = support_dir / "visual_support.jsonl"
        source_paths.extend([observation_path, support_path])
        supports = {row["candidate_id"]: row for row in read_jsonl(support_path)}
        observations = read_jsonl(observation_path)
        if len(observations) != 1400 or any(row.get("status") != "ok" for row in observations):
            raise RuntimeError(f"{arm}: expected 1400 complete observations")
        for row in observations:
            context = contexts[row["context_id"]]
            key = f"{arm}:{row['context_id']}:{row['candidate_id']}"
            visual = supports.get(row["candidate_id"])
            if not visual or visual.get("status") != "ok":
                raise RuntimeError(f"{arm}: missing visual support for {row['candidate_id']}")
            visual_by_key[key] = visual["label"]
            work.append({
                "key": key, "arm": arm, "context_id": row["context_id"],
                "staging_image_id": row["staging_image_id"],
                "candidate_id": row["candidate_id"],
                "premise": context["initial_caption"], "hypothesis": row["observed_text"],
            })
    payload = {
        "implementation": IMPLEMENTATION,
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths},
        "model": args.model, "revision": args.revision, "batch_size": args.batch_size,
        "arms": [name for name, _, _ in args.arm],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "entailments.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different R1 NLI fingerprint")
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    sentinel = {"NO_NEW_FACT", "NO_VISIBLE_FACT"}
    for row in work:
        if row["key"] not in existing and row["hypothesis"].strip().upper() in sentinel:
            existing[row["key"]] = {
                **{key: row[key] for key in ("key", "arm", "context_id",
                                             "staging_image_id", "candidate_id")},
                "label": "UNINFORMATIVE", "probabilities": {}, "status": "ok",
            }
    pending = [row for row in work if row["key"] not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "automated independent NLI sensitivity; not human gold",
        "expected": len(work), "completed": len(existing),
        "status": "running" if pending else "complete",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
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
                [row["premise"] for row in chunk],
                [row["hypothesis"] for row in chunk], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits.float().softmax(-1).cpu()
            for row, distribution in zip(chunk, probabilities):
                index = int(distribution.argmax())
                existing[row["key"]] = {
                    **{key: row[key] for key in ("key", "arm", "context_id",
                                                 "staging_image_id", "candidate_id")},
                    "label": id2label[index],
                    "probabilities": {id2label[i]: float(value)
                                      for i, value in enumerate(distribution)},
                    "status": "ok",
                }
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            runtime.update({
                "completed": len(existing), "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            })
            runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
            print(f"R1 independent NLI {len(existing)}/{len(work)}", flush=True)
    ordered = [existing[row["key"]] for row in work]
    failures = [row for row in ordered if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"R1 NLI incomplete: {len(failures)}")
    label_names = {row["label"] for row in ordered} - {"UNINFORMATIVE"}
    neutrals = {label for label in label_names if "NEUTRAL" in label}
    if len(neutrals) != 1:
        raise RuntimeError(f"could not identify neutral label: {sorted(label_names)}")
    neutral = next(iter(neutrals))
    grouped = defaultdict(list)
    for row in ordered:
        success = int(visual_by_key[row["key"]] == "SUPPORTED" and row["label"] == neutral)
        grouped[(row["arm"], row["context_id"])].append(success)
    summaries, oracle_vectors = {}, {}
    for arm, _, _ in args.arm:
        arm_rows = [row for row in ordered if row["arm"] == arm]
        context_ids = sorted({row["context_id"] for row in arm_rows})
        oracle = [int(any(grouped[(arm, context_id)])) for context_id in context_ids]
        clusters = [contexts[context_id]["staging_image_id"] for context_id in context_ids]
        oracle_vectors[arm] = (context_ids, oracle, clusters)
        summaries[arm] = {
            "outputs": len(arm_rows), "contexts": len(context_ids),
            "visual_counts": dict(Counter(visual_by_key[row["key"]] for row in arm_rows)),
            "nli_counts": dict(Counter(row["label"] for row in arm_rows)),
            "supported_and_novel_outputs": sum(
                visual_by_key[row["key"]] == "SUPPORTED" and row["label"] == neutral
                for row in arm_rows),
            "residual_oracle": bootstrap_cluster(oracle, clusters, samples=10000, seed=2271),
        }
    comparisons = {}
    names = [name for name, _, _ in args.arm]
    for left, right in ((names[1], names[0]), (names[3], names[2]),
                        (names[0], names[2]), (names[1], names[3])):
        left_ids, left_values, clusters = oracle_vectors[left]
        right_ids, right_values, _ = oracle_vectors[right]
        if left_ids != right_ids:
            raise RuntimeError("R1 arms do not use identical contexts")
        comparisons[f"{left}_minus_{right}"] = paired_bootstrap(
            left_values, right_values, clusters, samples=10000, seed=2271)
    report = {
        "status": "provisional_not_evidence",
        "warning": "Automated SmolVLM support plus DeBERTa novelty; not human fact adjudication.",
        "neutral_label": neutral, "arms": summaries,
        "paired_residual_oracle_differences": comparisons,
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
