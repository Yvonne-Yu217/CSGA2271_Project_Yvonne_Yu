"""Provisional model-assisted E1/E2 screen; explicitly not human gold evidence."""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from acquisition.observe import (DEFAULT_MODEL, DEFAULT_REVISION, atomic_jsonl,
                                 crop_candidate, read_jsonl)


LABELS = ("NEW_CORRECT", "REPEATED", "INVALID", "UNINFORMATIVE")
PROMPT = """Existing caption: {caption}
Candidate observation: {observation}
Using only the supplied crop, classify the candidate observation:
NEW_CORRECT = directly visible and adds at least one fact not entailed by the existing caption;
REPEATED = directly visible but only repeats facts already covered;
INVALID = false, unsupported, inferred, or contradicts the crop;
UNINFORMATIVE = no concrete recognizable fact.
Reply with exactly one label: NEW_CORRECT, REPEATED, INVALID, or UNINFORMATIVE."""


def parse_label(text):
    normalized = text.strip().upper().strip(".:-").replace(" ", "_")
    return normalized if normalized in LABELS else None


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--context-proposal-output", type=Path, required=True)
    parser.add_argument("--coordinate-planner-output", type=Path, required=True)
    parser.add_argument("--montage-planner-output", type=Path, required=True)
    parser.add_argument("--clip-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    input_paths = [
        args.staging / "staging_images.jsonl", args.staging / "automatic_candidates.jsonl",
        args.observer_output / "observations.jsonl",
        args.context_proposal_output / "context_proposals.jsonl",
        args.coordinate_planner_output / "choices.jsonl", args.montage_planner_output / "choices.jsonl",
        args.clip_output / "scores.jsonl",
    ]
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in input_paths},
               "model": args.model, "revision": args.revision, "prompt": PROMPT,
               "images": args.images, "batch_size": args.batch_size,
               "implementation": "provisional-screen-v1"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    proposals = {row["staging_image_id"]: row
                 for row in read_jsonl(args.context_proposal_output / "context_proposals.jsonl")}
    selected_ids = sorted(set(images) & set(proposals))[:args.images]
    candidates_by_image = defaultdict(list)
    candidates = {}
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        candidates[row["candidate_id"]] = row
        if row["staging_image_id"] in selected_ids:
            candidates_by_image[row["staging_image_id"]].append(row)
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    work = []
    for image_id in selected_ids:
        proposal = proposals[image_id]
        context_texts = {"natural": proposal["reference_caption"], **proposal["proposals"]}
        for kind, caption in context_texts.items():
            for candidate in sorted(candidates_by_image[image_id], key=lambda row: row["candidate_id"]):
                if candidate["kind"] == "stop":
                    continue
                observation = observations[candidate["candidate_id"]]
                work.append({"key": f"{image_id}:{kind}:{candidate['candidate_id']}",
                             "staging_image_id": image_id, "context_kind": kind,
                             "caption": caption, "candidate": candidate,
                             "observation": observation["observed_text"]})
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "judgments.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different provisional screen")
    pending = [row for row in work if row["key"] not in existing and
               row["observation"] != "NO_VISIBLE_FACT"]
    for row in work:
        if row["observation"] == "NO_VISIBLE_FACT" and row["key"] not in existing:
            existing[row["key"]] = {
                "key": row["key"], "staging_image_id": row["staging_image_id"],
                "context_kind": row["context_kind"], "candidate_id": row["candidate"]["candidate_id"],
                "label": "UNINFORMATIVE", "raw_output": "deterministic:NO_VISIBLE_FACT",
                "status": "ok", "output_tokens": 0, "latency_ms": 0,
            }
    metadata = {"scope": "provisional same-model screen; not independent and not human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "images": len(selected_ids), "expected_judgments": len(work),
                "completed": len(existing), "status": "running" if pending else "complete",
                "slurm_job_id": os.getenv("SLURM_JOB_ID")}
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
            crops, texts = [], []
            for row in chunk:
                crops.append(crop_candidate(args.staging, images[row["staging_image_id"]],
                                            row["candidate"]))
                prompt = PROMPT.format(caption=row["caption"], observation=row["observation"])
                conversation = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, images=crops, padding=True, return_tensors="pt").to("cuda")
            batch_started = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            latency = round(1000 * (time.time() - batch_started) / len(chunk))
            for row, raw, output_ids in zip(chunk, decoded, trimmed):
                label = parse_label(raw)
                existing[row["key"]] = {
                    "key": row["key"], "staging_image_id": row["staging_image_id"],
                    "context_kind": row["context_kind"],
                    "candidate_id": row["candidate"]["candidate_id"],
                    "label": label, "raw_output": raw.strip(),
                    "status": "ok" if label else "parse_failed",
                    "output_tokens": int(output_ids.numel()), "latency_ms": latency,
                }
            for crop in crops:
                crop.close()
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"screen {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    rows = list(existing.values())
    if len(rows) != len(work) or any(row["status"] != "ok" for row in rows):
        raise RuntimeError("provisional judgment cache is incomplete or unparsed")
    lookup = {(row["staging_image_id"], row["context_kind"], row["candidate_id"]): row["label"]
              for row in rows}
    coordinate = {row["context_id"]: row for row in read_jsonl(
        args.coordinate_planner_output / "choices.jsonl")}
    montage = {row["context_id"]: row for row in read_jsonl(
        args.montage_planner_output / "choices.jsonl")}
    clip_rows = read_jsonl(args.clip_output / "scores.jsonl")
    clip_by_context = defaultdict(list)
    for row in clip_rows:
        clip_by_context[row["context_id"]].append(row)
    per_image = []
    score_vectors = defaultdict(dict)
    for image_id in selected_ids:
        proposal = proposals[image_id]
        natural_id = proposal["reference_context_id"]
        nonstops = [row for row in candidates_by_image[image_id] if row["kind"] != "stop"]
        stop = next(row for row in candidates_by_image[image_id] if row["kind"] == "stop")
        for kind in ("natural", "enriched", "paraphrase", "saturated"):
            for candidate in nonstops:
                label = lookup[(image_id, kind, candidate["candidate_id"])]
                score_vectors[(image_id, kind)][candidate["candidate_id"]] = int(label == "NEW_CORRECT")
            score_vectors[(image_id, kind)][stop["candidate_id"]] = int(
                not any(score_vectors[(image_id, kind)].values()))
        chosen = {
            "coordinate_planner": coordinate[natural_id]["candidate_id"],
            "montage_planner": montage[natural_id]["candidate_id"],
            "clip_inverse": max(clip_by_context[natural_id],
                                key=lambda row: row["inverse_cosine_score"])["candidate_id"],
            "largest_area": max(nonstops, key=lambda row: row["pixel_cost"])["candidate_id"],
            "stop": stop["candidate_id"],
        }
        labels = {candidate["candidate_id"]: lookup[(image_id, "natural", candidate["candidate_id"])]
                  for candidate in nonstops}
        for method, candidate_id in chosen.items():
            label = "UNINFORMATIVE" if candidate_id == stop["candidate_id"] else labels[candidate_id]
            per_image.append({"staging_image_id": image_id, "method": method,
                              "candidate_id": candidate_id, "label": label,
                              "new_correct": int(label == "NEW_CORRECT"),
                              "invalid": int(label == "INVALID")})
        oracle_candidates = [candidate_id for candidate_id, label in labels.items()
                             if label == "NEW_CORRECT"]
        per_image.append({"staging_image_id": image_id, "method": "recognition_oracle",
                          "candidate_id": min(oracle_candidates,
                                              key=lambda value: candidates[value]["pixel_cost"])
                          if oracle_candidates else stop["candidate_id"],
                          "label": "NEW_CORRECT" if oracle_candidates else "UNINFORMATIVE",
                          "new_correct": int(bool(oracle_candidates)), "invalid": 0})
    method_summary = {}
    for method in sorted({row["method"] for row in per_image}):
        subset = [row for row in per_image if row["method"] == method]
        method_summary[method] = {"images": len(subset),
                                  "new_correct_rate": mean([row["new_correct"] for row in subset]),
                                  "invalid_rate": mean([row["invalid"] for row in subset]),
                                  "label_counts": dict(Counter(row["label"] for row in subset))}
    necessity, switches, saturated_stop = [], [], []
    for image_id in selected_ids:
        vectors = {kind: score_vectors[(image_id, kind)]
                   for kind in ("natural", "enriched", "paraphrase", "saturated")}
        candidate_ids = list(next(iter(vectors.values())))
        adaptive = mean([max(vector.values()) for vector in vectors.values()])
        static = max(mean([vectors[kind][candidate_id] for kind in vectors])
                     for candidate_id in candidate_ids)
        necessity.append(adaptive - static)
        natural_top = {candidate_id for candidate_id, value in vectors["natural"].items()
                       if value == max(vectors["natural"].values())}
        saturated_top = {candidate_id for candidate_id, value in vectors["saturated"].items()
                         if value == max(vectors["saturated"].values())}
        switches.append(int(not bool(natural_top & saturated_top)))
        stop_id = next(row["candidate_id"] for row in candidates_by_image[image_id]
                       if row["kind"] == "stop")
        saturated_stop.append(int(vectors["saturated"][stop_id] == 1))
    report = {
        "status": "provisional_not_evidence",
        "warning": ("The observer and judge use the same Qwen family; generated context proposals are "
                    "unreviewed. This screen may be circular and cannot pass E0/E1/E2."),
        "fingerprint": fingerprint, "images": len(selected_ids),
        "judgment_counts": dict(Counter(row["label"] for row in rows)),
        "e1_natural": {"methods": method_summary,
                       "oracle_minus_coordinate": method_summary["recognition_oracle"]["new_correct_rate"] -
                       method_summary["coordinate_planner"]["new_correct_rate"],
                       "oracle_minus_montage": method_summary["recognition_oracle"]["new_correct_rate"] -
                       method_summary["montage_planner"]["new_correct_rate"]},
        "e2_oracle_data_screen": {"caption_necessity_gap_mean": mean(necessity),
                                  "natural_to_saturated_material_switch_rate": mean(switches),
                                  "saturated_stop_oracle_rate": mean(saturated_stop)},
        "per_image_method_rows": per_image,
    }
    (args.output / "provisional_report.json").write_text(json.dumps(report, indent=2,
                                                                      sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "images", "judgment_counts",
                                                   "e1_natural", "e2_oracle_data_screen")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
