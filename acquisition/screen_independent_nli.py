"""Independent text-NLI sensitivity check for the provisional natural E1 screen."""
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


DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--visual-support-output", type=Path, required=True)
    parser.add_argument("--selection-report-output", type=Path, required=True)
    parser.add_argument("--qwen-entailment-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    contexts = {row["context_id"]: row
                for row in read_jsonl(args.staging / "natural_contexts.jsonl")}
    candidates, candidates_by_image = {}, defaultdict(list)
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        candidates[row["candidate_id"]] = row
        candidates_by_image[row["staging_image_id"]].append(row)
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    visual = {row["candidate_id"]: row["label"] for row in read_jsonl(
        args.visual_support_output / "visual_support.jsonl")}
    work = []
    for context in contexts.values():
        for candidate in candidates_by_image[context["staging_image_id"]]:
            if candidate["kind"] == "stop":
                continue
            work.append({"key": context["context_id"] + ":" + candidate["candidate_id"],
                         "context_id": context["context_id"],
                         "staging_image_id": context["staging_image_id"],
                         "candidate_id": candidate["candidate_id"],
                         "premise": context["initial_caption"],
                         "hypothesis": observations[candidate["candidate_id"]]["observed_text"]})
    sources = [args.staging / "natural_contexts.jsonl",
               args.staging / "automatic_candidates.jsonl",
               args.observer_output / "observations.jsonl"]
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "batch_size": args.batch_size,
               "implementation": "independent-deberta-nli-v1"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "entailments.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to another independent NLI run")
    pending = [row for row in work if row["key"] not in existing and
               row["hypothesis"] != "NO_VISIBLE_FACT"]
    for row in work:
        if row["key"] not in existing and row["hypothesis"] == "NO_VISIBLE_FACT":
            existing[row["key"]] = {"key": row["key"], "context_id": row["context_id"],
                                    "staging_image_id": row["staging_image_id"],
                                    "candidate_id": row["candidate_id"],
                                    "label": "UNINFORMATIVE", "probabilities": {},
                                    "status": "ok"}
    metadata = {"scope": "independent text NLI sensitivity; visual support remains same-family",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "expected": len(work), "completed": len(existing),
                "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    if not pending and old_metadata:
        metadata = {**metadata, **old_metadata, "completed": len(existing), "status": "complete"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  local_files_only=not args.allow_model_download)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision,
            local_files_only=not args.allow_model_download, dtype=torch.float16).to("cuda").eval()
        id2label = {int(key): value.upper() for key, value in model.config.id2label.items()}
        metadata["id2label"] = id2label
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for offset in range(0, len(pending), args.batch_size):
            chunk = pending[offset:offset + args.batch_size]
            inputs = tokenizer([row["premise"] for row in chunk],
                               [row["hypothesis"] for row in chunk], padding=True,
                               truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                probabilities = model(**inputs).logits.float().softmax(-1).cpu()
            for row, distribution in zip(chunk, probabilities):
                index = int(distribution.argmax())
                existing[row["key"]] = {
                    "key": row["key"], "context_id": row["context_id"],
                    "staging_image_id": row["staging_image_id"],
                    "candidate_id": row["candidate_id"], "label": id2label[index],
                    "probabilities": {id2label[i]: float(value)
                                      for i, value in enumerate(distribution)}, "status": "ok"}
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"independent NLI {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = list(existing.values())
    if len(rows) != len(work) or any(row["status"] != "ok" for row in rows):
        raise RuntimeError("independent NLI cache incomplete")
    label_names = {row["label"] for row in rows} - {"UNINFORMATIVE"}
    neutral_labels = {label for label in label_names if "NEUTRAL" in label}
    if len(neutral_labels) != 1:
        raise RuntimeError(f"could not identify unique neutral label: {sorted(label_names)}")
    neutral = next(iter(neutral_labels))
    nli = {(row["context_id"], row["candidate_id"]): row["label"] for row in rows}
    selection_report = json.loads(
        (args.selection_report_output / "provisional_report.json").read_text())
    selected = {(row["context_id"], row["method"]): row["candidate_id"]
                for row in selection_report["context_rows"] if row["candidate_id"] is not None}
    methods = ("montage_planner", "coordinate_planner", "largest_area",
               "clip_inverse", "clip_inverse_cost_matched")
    result_rows = []
    for context_id, context in contexts.items():
        image_id = context["staging_image_id"]
        nonstops = [row for row in candidates_by_image[image_id] if row["kind"] != "stop"]
        stop = next(row for row in candidates_by_image[image_id] if row["kind"] == "stop")
        values = {row["candidate_id"]: int(
            visual[row["candidate_id"]] == "SUPPORTED" and
            nli[(context_id, row["candidate_id"])] == neutral) for row in nonstops}
        values[stop["candidate_id"]] = int(not any(values.values()))
        for method in methods:
            candidate_id = selected[(context_id, method)]
            result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                                "method": method, "new_correct": values[candidate_id]})
        budget = candidates[selected[(context_id, "montage_planner")]]["pixel_cost"]
        budget_ids = [candidate_id for candidate_id in values
                      if candidates[candidate_id]["pixel_cost"] <= budget]
        oracle_id = min(budget_ids, key=lambda cid: (-values[cid], candidates[cid]["pixel_cost"], cid))
        result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                            "method": "recognition_oracle_cost_matched",
                            "new_correct": values[oracle_id]})
        result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                            "method": "random_cost_matched_to_montage",
                            "new_correct": sum(values[cid] for cid in budget_ids) / len(budget_ids)})
    summaries = {}
    for method in sorted({row["method"] for row in result_rows}):
        subset = [row for row in result_rows if row["method"] == method]
        summaries[method] = bootstrap_cluster(
            [row["new_correct"] for row in subset],
            [row["staging_image_id"] for row in subset], samples=10000, seed=2271)
    by_key = {(row["context_id"], row["method"]): row["new_correct"] for row in result_rows}
    context_ids = sorted(contexts)
    clusters = [contexts[context_id]["staging_image_id"] for context_id in context_ids]
    gaps = {method: paired_bootstrap(
        [by_key[(context_id, "recognition_oracle_cost_matched")] for context_id in context_ids],
        [by_key[(context_id, method)] for context_id in context_ids], clusters,
        samples=10000, seed=2271) for method in methods}
    agreement = None
    if args.qwen_entailment_output:
        qwen = {(row["context_id"], row["candidate_id"]): row["label"] for row in read_jsonl(
            args.qwen_entailment_output / "entailments.jsonl")}
        paired = [(row, qwen[(row["context_id"], row["candidate_id"])]) for row in rows]
        agreement = {
            "all_label_binary_new_agreement": sum(
                (row["label"] == neutral) == (qwen_label == "NOT_ENTAILED")
                for row, qwen_label in paired) / len(paired),
            "deberta_new_rate": sum(row["label"] == neutral for row, _ in paired) / len(paired),
            "qwen_new_rate": sum(label == "NOT_ENTAILED" for _, label in paired) / len(paired),
        }
    report = {"status": "provisional_not_evidence",
              "warning": "Independent text NLI only; crop visual support is still same-family and unreviewed.",
              "images": len({row["staging_image_id"] for row in contexts.values()}),
              "contexts": len(contexts), "label_counts": dict(Counter(row["label"] for row in rows)),
              "qwen_comparison": agreement, "methods": summaries,
              "oracle_minus_baseline": gaps, "context_rows": result_rows}
    (args.output / "provisional_report.json").write_text(json.dumps(report, indent=2,
                                                                      sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "images", "contexts", "label_counts",
                                                   "qwen_comparison", "methods",
                                                   "oracle_minus_baseline")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
