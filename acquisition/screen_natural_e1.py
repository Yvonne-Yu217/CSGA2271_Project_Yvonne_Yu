"""100-image/500-caption provisional E1 screen with separated support and entailment."""
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
from acquisition.observe import DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl, read_jsonl
from acquisition.screen_entailment import LABELS, PROMPT, parse_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--visual-support-output", type=Path, required=True)
    parser.add_argument("--coordinate-planner-output", type=Path, required=True)
    parser.add_argument("--montage-planner-output", type=Path, required=True)
    parser.add_argument("--clip-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-entailments-from", type=Path,
                        help="Reuse judgments only when every expected key matches")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts = read_jsonl(args.staging / "natural_contexts.jsonl")
    candidates, candidates_by_image = {}, defaultdict(list)
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        candidates[row["candidate_id"]] = row
        candidates_by_image[row["staging_image_id"]].append(row)
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    visual = {row["candidate_id"]: row["label"] for row in read_jsonl(
        args.visual_support_output / "visual_support.jsonl")}
    if set(visual) != {cid for cid, row in candidates.items() if row["kind"] != "stop"}:
        raise RuntimeError("visual support does not cover every non-STOP candidate")
    work = []
    for context in contexts:
        for candidate in candidates_by_image[context["staging_image_id"]]:
            if candidate["kind"] == "stop":
                continue
            observation = observations[candidate["candidate_id"]]["observed_text"]
            work.append({"key": context["context_id"] + ":" + candidate["candidate_id"],
                         "context_id": context["context_id"],
                         "staging_image_id": context["staging_image_id"],
                         "candidate_id": candidate["candidate_id"],
                         "caption": context["initial_caption"], "observation": observation})
    sources = [args.staging / "natural_contexts.jsonl",
               args.staging / "automatic_candidates.jsonl",
               args.observer_output / "observations.jsonl",
               args.visual_support_output / "visual_support.jsonl"]
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "prompt": PROMPT,
               "batch_size": args.batch_size, "implementation": "provisional-natural-e1-v1"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "entailments.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different natural E1 screen")
    expected_keys = {row["key"] for row in work}
    if not existing and args.reuse_entailments_from:
        reusable_metadata = json.loads(
            (args.reuse_entailments_from / "runtime.json").read_text())
        if reusable_metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("reusable natural E1 cache has a different inference fingerprint")
        reusable = read_jsonl(args.reuse_entailments_from / "entailments.jsonl")
        reusable_by_key = {row.get("key"): row for row in reusable}
        if (set(reusable_by_key) != expected_keys or len(reusable_by_key) != len(reusable) or
                any(row.get("status") != "ok" or row.get("label") not in LABELS
                    for row in reusable)):
            raise RuntimeError("reusable natural E1 entailments do not match expected work")
        existing = reusable_by_key
    pending = []
    for row in work:
        if row["key"] in existing:
            continue
        if row["observation"] == "NO_VISIBLE_FACT":
            existing[row["key"]] = {"key": row["key"], "context_id": row["context_id"],
                                    "staging_image_id": row["staging_image_id"],
                                    "candidate_id": row["candidate_id"], "label": "NOT_ENTAILED",
                                    "raw_output": "skipped:NO_VISIBLE_FACT", "status": "ok"}
        else:
            pending.append(row)
    metadata = {"scope": "provisional same-model screen; not independent or human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "images": len(images), "contexts": len(contexts), "expected": len(work),
                "completed": len(existing), "status": "running" if pending else "complete",
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "baseline_scores_sha256": hashlib.sha256(
                    (args.clip_output / "scores.jsonl").read_bytes()).hexdigest()}
    if not pending and old_metadata:
        metadata = {**metadata, **old_metadata, "completed": len(existing), "status": "complete"}
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
            for row in chunk:
                prompt = PROMPT.format(caption=row["caption"], observation=row["observation"])
                conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            for row, raw in zip(chunk, decoded):
                label = parse_label(raw)
                existing[row["key"]] = {"key": row["key"], "context_id": row["context_id"],
                                        "staging_image_id": row["staging_image_id"],
                                        "candidate_id": row["candidate_id"], "label": label,
                                        "raw_output": raw.strip(),
                                        "status": "ok" if label else "parse_failed"}
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"natural E1 {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = list(existing.values())
    if len(rows) != len(work) or any(row["status"] != "ok" for row in rows):
        raise RuntimeError("natural E1 entailment cache incomplete or unparsed")
    entailment = {(row["context_id"], row["candidate_id"]): row["label"] for row in rows}
    coordinate = {row["context_id"]: row for row in read_jsonl(
        args.coordinate_planner_output / "choices.jsonl")}
    montage = {row["context_id"]: row for row in read_jsonl(
        args.montage_planner_output / "choices.jsonl")}
    clip_by_context = defaultdict(list)
    for row in read_jsonl(args.clip_output / "scores.jsonl"):
        clip_by_context[row["context_id"]].append(row)
    result_rows = []
    for context in contexts:
        image_id = context["staging_image_id"]
        nonstops = [row for row in candidates_by_image[image_id] if row["kind"] != "stop"]
        stop = next(row for row in candidates_by_image[image_id] if row["kind"] == "stop")
        values = {row["candidate_id"]: int(
            visual[row["candidate_id"]] == "SUPPORTED" and
            entailment[(context["context_id"], row["candidate_id"])] == "NOT_ENTAILED")
                  for row in nonstops}
        values[stop["candidate_id"]] = int(not any(values.values()))
        selected = {
            "coordinate_planner": coordinate[context["context_id"]]["candidate_id"],
            "montage_planner": montage[context["context_id"]]["candidate_id"],
            "clip_inverse": max(clip_by_context[context["context_id"]],
                                key=lambda row: row["inverse_cosine_score"])["candidate_id"],
            "largest_area": max(nonstops, key=lambda row: row["pixel_cost"])["candidate_id"],
            "stop": stop["candidate_id"],
            "recognition_oracle": max(values, key=values.get),
        }
        montage_budget = candidates[selected["montage_planner"]]["pixel_cost"]
        budget_ids = [candidate_id for candidate_id in values
                      if candidates[candidate_id]["pixel_cost"] <= montage_budget]
        selected["recognition_oracle_cost_matched"] = min(
            budget_ids, key=lambda candidate_id: (-values[candidate_id],
                                                  candidates[candidate_id]["pixel_cost"],
                                                  candidate_id))
        eligible_clip = [row for row in clip_by_context[context["context_id"]]
                         if candidates[row["candidate_id"]]["pixel_cost"] <= montage_budget]
        selected["clip_inverse_cost_matched"] = max(
            eligible_clip, key=lambda row: row["inverse_cosine_score"])["candidate_id"]
        for method, candidate_id in selected.items():
            visual_label = visual.get(candidate_id, "STOP")
            result_rows.append({"context_id": context["context_id"], "staging_image_id": image_id,
                                "method": method, "candidate_id": candidate_id,
                                "new_correct": values[candidate_id],
                                "strict_visual_error": int(visual_label in {"PARTIAL", "INVALID"}),
                                "pixel_cost": candidates[candidate_id]["pixel_cost"]})
        result_rows.append({"context_id": context["context_id"], "staging_image_id": image_id,
                            "method": "random_expected", "candidate_id": None,
                            "new_correct": sum(values.values()) / len(values),
                            "strict_visual_error": mean([
                                int(visual.get(cid, "STOP") in {"PARTIAL", "INVALID"}) for cid in values]),
                            "pixel_cost": mean([candidates[cid]["pixel_cost"] for cid in values])})
        result_rows.append({"context_id": context["context_id"], "staging_image_id": image_id,
                            "method": "random_cost_matched_to_montage", "candidate_id": None,
                            "new_correct": mean([values[cid] for cid in budget_ids]),
                            "strict_visual_error": mean([
                                int(visual.get(cid, "STOP") in {"PARTIAL", "INVALID"})
                                for cid in budget_ids]),
                            "pixel_cost": mean([candidates[cid]["pixel_cost"] for cid in budget_ids])})
    summaries = {}
    for method in sorted({row["method"] for row in result_rows}):
        subset = [row for row in result_rows if row["method"] == method]
        summaries[method] = {
            metric: bootstrap_cluster([row[metric] for row in subset],
                                      [row["staging_image_id"] for row in subset],
                                      samples=10000, seed=2271)
            for metric in ("new_correct", "strict_visual_error", "pixel_cost")}
    by_key = {(row["context_id"], row["method"]): row for row in result_rows}
    context_ids = sorted(row["context_id"] for row in contexts)
    clusters = [next(row["staging_image_id"] for row in contexts
                     if row["context_id"] == context_id) for context_id in context_ids]
    gaps = {}
    for baseline in ("montage_planner", "coordinate_planner", "clip_inverse", "largest_area",
                     "clip_inverse_cost_matched", "random_cost_matched_to_montage"):
        gaps[baseline] = paired_bootstrap(
            [by_key[(context_id, "recognition_oracle")]["new_correct"] for context_id in context_ids],
            [by_key[(context_id, baseline)]["new_correct"] for context_id in context_ids],
            clusters, samples=10000, seed=2271)
    montage_gains = {}
    for baseline in ("coordinate_planner", "clip_inverse", "largest_area",
                     "clip_inverse_cost_matched", "random_cost_matched_to_montage"):
        montage_gains[baseline] = paired_bootstrap(
            [by_key[(context_id, "montage_planner")]["new_correct"] for context_id in context_ids],
            [by_key[(context_id, baseline)]["new_correct"] for context_id in context_ids],
            clusters, samples=10000, seed=2271)
    report = {"status": "provisional_not_evidence",
              "warning": ("Observer, visual-support judge, and entailment judge share the Qwen family. "
                          "No human fact adjudication; systematic self-confirmation is likely."),
              "images": len(images), "contexts": len(contexts),
              "semantic_baseline": json.loads((args.clip_output / "runtime.json").read_text()),
              "visual_support_counts": dict(Counter(visual.values())),
              "entailment_counts": dict(Counter(row["label"] for row in rows)),
              "methods": summaries, "oracle_minus_baseline": gaps,
              "montage_minus_baseline": montage_gains,
              "context_rows": result_rows}
    (args.output / "provisional_report.json").write_text(json.dumps(report, indent=2,
                                                                      sort_keys=True) + "\n")
    compact = {"status": report["status"], "images": report["images"],
               "contexts": report["contexts"], "visual_support_counts": report["visual_support_counts"],
               "entailment_counts": report["entailment_counts"],
               "new_correct": {method: values["new_correct"] for method, values in summaries.items()},
               "oracle_minus_baseline": gaps}
    compact["montage_minus_baseline"] = montage_gains
    print(json.dumps(compact, indent=2, sort_keys=True))


def mean(values):
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
