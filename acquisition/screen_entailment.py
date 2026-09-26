"""Text-entailment correction for the provisional direction screen."""
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


LABELS = ("ENTAILED", "NOT_ENTAILED", "CONTRADICTED")
PROMPT = """Premise (existing image caption): {caption}
Hypothesis (candidate observation): {observation}
Judge only the semantic relationship. ENTAILED means every concrete fact in the hypothesis is
already stated or necessarily implied by the premise. NOT_ENTAILED means at least one concrete
fact is additional. CONTRADICTED means they conflict. Do not use outside knowledge.
Reply with exactly ENTAILED, NOT_ENTAILED, or CONTRADICTED."""


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
    parser.add_argument("--joint-screen-output", type=Path, required=True)
    parser.add_argument("--visual-support-output", type=Path)
    parser.add_argument("--coordinate-planner-output", type=Path, required=True)
    parser.add_argument("--montage-planner-output", type=Path, required=True)
    parser.add_argument("--clip-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-entailments-from", type=Path,
                        help="Reuse judgments only when all expected keys match exactly")
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--constructed-contexts", action="store_true",
                        help="Use action-aligned covered-one and observer-saturated contexts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    proposals = {row["staging_image_id"]: row
                 for row in read_jsonl(args.context_proposal_output / "context_proposals.jsonl")}
    selected_ids = sorted(proposals)[:args.images]
    candidates_by_image = defaultdict(list)
    candidates = {}
    for row in read_jsonl(args.staging / "automatic_candidates.jsonl"):
        candidates[row["candidate_id"]] = row
        if row["staging_image_id"] in selected_ids:
            candidates_by_image[row["staging_image_id"]].append(row)
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    montage_choices = {row["context_id"]: row for row in read_jsonl(
        args.montage_planner_output / "choices.jsonl")}
    joint = read_jsonl(args.joint_screen_output / "judgments.jsonl")
    joint_by_candidate = defaultdict(list)
    for row in joint:
        joint_by_candidate[row["candidate_id"]].append(row["label"])
    visually_invalid = {candidate_id for candidate_id, labels in joint_by_candidate.items()
                        if labels.count("INVALID") >= 2}
    visual_support = None
    if args.visual_support_output:
        visual_support = {row["candidate_id"]: row["label"] for row in read_jsonl(
            args.visual_support_output / "visual_support.jsonl")}
        visually_invalid = {candidate_id for candidate_id, label in visual_support.items()
                            if label != "SUPPORTED"}
    work = []
    for image_id in selected_ids:
        proposal = proposals[image_id]
        if args.constructed_contexts:
            chosen_id = montage_choices[proposal["reference_context_id"]]["candidate_id"]
            supported_texts = [observations[row["candidate_id"]]["observed_text"]
                               for row in sorted(candidates_by_image[image_id],
                                                 key=lambda value: value["candidate_id"])
                               if row["kind"] != "stop" and
                               row["candidate_id"] not in visually_invalid and
                               observations[row["candidate_id"]]["observed_text"] != "NO_VISIBLE_FACT"]
            context_texts = {
                "natural": proposal["reference_caption"],
                "covered_one": proposal["reference_caption"] + " " +
                               observations[chosen_id]["observed_text"],
                "paraphrase": proposal["proposals"]["paraphrase"],
                "observer_saturated": proposal["reference_caption"] + " " + " ".join(supported_texts),
            }
        else:
            context_texts = {"natural": proposal["reference_caption"], **proposal["proposals"]}
        for kind, caption in context_texts.items():
            for candidate in sorted(candidates_by_image[image_id], key=lambda row: row["candidate_id"]):
                if candidate["kind"] == "stop":
                    continue
                observation = observations[candidate["candidate_id"]]["observed_text"]
                work.append({"key": f"{image_id}:{kind}:{candidate['candidate_id']}",
                             "staging_image_id": image_id, "context_kind": kind,
                             "caption": caption, "candidate_id": candidate["candidate_id"],
                             "observation": observation})
    sources = [args.staging / "automatic_candidates.jsonl",
               args.observer_output / "observations.jsonl",
               args.context_proposal_output / "context_proposals.jsonl",
               args.joint_screen_output / "judgments.jsonl"]
    if args.visual_support_output:
        sources.append(args.visual_support_output / "visual_support.jsonl")
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "prompt": PROMPT,
               "images": args.images, "batch_size": args.batch_size,
               "constructed_contexts": args.constructed_contexts,
               "implementation": "provisional-entailment-v1"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "entailments.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different entailment screen")
    expected_keys = {row["key"] for row in work}
    if not existing and args.reuse_entailments_from:
        reusable_metadata = json.loads(
            (args.reuse_entailments_from / "runtime.json").read_text())
        if reusable_metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("reusable entailment cache has a different inference fingerprint")
        reusable = read_jsonl(args.reuse_entailments_from / "entailments.jsonl")
        reusable_by_key = {row.get("key"): row for row in reusable}
        if (set(reusable_by_key) != expected_keys or len(reusable_by_key) != len(reusable) or
                any(row.get("status") != "ok" or row.get("label") not in LABELS
                    for row in reusable)):
            raise RuntimeError("reusable entailment cache does not match expected work exactly")
        existing = reusable_by_key
    pending = [row for row in work if row["key"] not in existing and
               row["observation"] != "NO_VISIBLE_FACT"]
    for row in work:
        if row["observation"] == "NO_VISIBLE_FACT" and row["key"] not in existing:
            existing[row["key"]] = {"key": row["key"], "staging_image_id": row["staging_image_id"],
                                    "context_kind": row["context_kind"],
                                    "candidate_id": row["candidate_id"], "label": "NOT_ENTAILED",
                                    "raw_output": "skipped:NO_VISIBLE_FACT", "status": "ok"}
    metadata = {"scope": "provisional same-model text entailment; not human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "images": len(selected_ids), "expected": len(work), "completed": len(existing),
                "status": "running" if pending else "complete", "slurm_job_id": os.getenv("SLURM_JOB_ID")}
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
                existing[row["key"]] = {"key": row["key"],
                                        "staging_image_id": row["staging_image_id"],
                                        "context_kind": row["context_kind"],
                                        "candidate_id": row["candidate_id"], "label": label,
                                        "raw_output": raw.strip(),
                                        "status": "ok" if label else "parse_failed"}
            atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
            metadata.update({"completed": len(existing), "elapsed_seconds": time.time() - started,
                             "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"entailment {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = list(existing.values())
    if len(rows) != len(work) or any(row["status"] != "ok" for row in rows):
        raise RuntimeError("entailment cache incomplete or unparsed")
    entailment = {(row["staging_image_id"], row["context_kind"], row["candidate_id"]): row["label"]
                  for row in rows}
    coordinate = {row["context_id"]: row for row in read_jsonl(
        args.coordinate_planner_output / "choices.jsonl")}
    montage = {row["context_id"]: row for row in read_jsonl(
        args.montage_planner_output / "choices.jsonl")}
    clip_by_context = defaultdict(list)
    for row in read_jsonl(args.clip_output / "scores.jsonl"):
        clip_by_context[row["context_id"]].append(row)
    vectors, per_image = {}, []
    for image_id in selected_ids:
        proposal = proposals[image_id]
        natural_id = proposal["reference_context_id"]
        nonstops = [row for row in candidates_by_image[image_id] if row["kind"] != "stop"]
        stop = next(row for row in candidates_by_image[image_id] if row["kind"] == "stop")
        kinds = (("natural", "covered_one", "paraphrase", "observer_saturated")
                 if args.constructed_contexts else
                 ("natural", "enriched", "paraphrase", "saturated"))
        for kind in kinds:
            vector = {}
            for candidate in nonstops:
                cid = candidate["candidate_id"]
                informative = observations[cid]["observed_text"] != "NO_VISIBLE_FACT"
                valid = informative and cid not in visually_invalid
                vector[cid] = int(valid and entailment[(image_id, kind, cid)] == "NOT_ENTAILED")
            vector[stop["candidate_id"]] = int(not any(vector.values()))
            vectors[(image_id, kind)] = vector
        chosen = {
            "coordinate_planner": coordinate[natural_id]["candidate_id"],
            "montage_planner": montage[natural_id]["candidate_id"],
            "clip_inverse": max(clip_by_context[natural_id],
                                key=lambda row: row["inverse_cosine_score"])["candidate_id"],
            "largest_area": max(nonstops, key=lambda row: row["pixel_cost"])["candidate_id"],
            "stop": stop["candidate_id"],
        }
        natural = vectors[(image_id, "natural")]
        for method, cid in chosen.items():
            per_image.append({"staging_image_id": image_id, "method": method,
                              "candidate_id": cid, "new_correct": natural[cid]})
        per_image.append({"staging_image_id": image_id, "method": "recognition_oracle",
                          "candidate_id": max(natural, key=natural.get),
                          "new_correct": max(natural.values())})
    methods = {method: mean([row["new_correct"] for row in per_image if row["method"] == method])
               for method in sorted({row["method"] for row in per_image})}
    method_intervals = {
        method: bootstrap_cluster(
            [row["new_correct"] for row in per_image if row["method"] == method],
            [row["staging_image_id"] for row in per_image if row["method"] == method],
            samples=10000, seed=2271)
        for method in methods
    }
    by_method_image = {(row["method"], row["staging_image_id"]): row["new_correct"]
                       for row in per_image}
    oracle_coordinate_ci = paired_bootstrap(
        [by_method_image[("recognition_oracle", image_id)] for image_id in selected_ids],
        [by_method_image[("coordinate_planner", image_id)] for image_id in selected_ids],
        selected_ids, samples=10000, seed=2271)
    oracle_montage_ci = paired_bootstrap(
        [by_method_image[("recognition_oracle", image_id)] for image_id in selected_ids],
        [by_method_image[("montage_planner", image_id)] for image_id in selected_ids],
        selected_ids, samples=10000, seed=2271)
    necessity, switches, saturated_stops, redundancy_losses = [], [], [], []
    for image_id in selected_ids:
        kinds = (("natural", "covered_one", "paraphrase", "observer_saturated")
                 if args.constructed_contexts else
                 ("natural", "enriched", "paraphrase", "saturated"))
        image_vectors = {kind: vectors[(image_id, kind)] for kind in kinds}
        ids = list(image_vectors["natural"])
        adaptive = mean([max(vector.values()) for vector in image_vectors.values()])
        static = max(mean([image_vectors[kind][cid] for kind in image_vectors]) for cid in ids)
        necessity.append(adaptive - static)
        natural_top = {cid for cid, value in image_vectors["natural"].items()
                       if value == max(image_vectors["natural"].values())}
        saturated_kind = "observer_saturated" if args.constructed_contexts else "saturated"
        saturated_top = {cid for cid, value in image_vectors[saturated_kind].items()
                         if value == max(image_vectors[saturated_kind].values())}
        switches.append(int(not bool(natural_top & saturated_top)))
        stop_id = next(row["candidate_id"] for row in candidates_by_image[image_id]
                       if row["kind"] == "stop")
        saturated_stops.append(image_vectors[saturated_kind][stop_id])
        if args.constructed_contexts:
            natural_id = proposals[image_id]["reference_context_id"]
            chosen_id = montage_choices[natural_id]["candidate_id"]
            redundancy_losses.append(image_vectors["natural"][chosen_id] -
                                     image_vectors["covered_one"][chosen_id])
    limitation = ("Same-family observer, visual-support judge, and entailment judge plus "
                  "unreviewed generated/constructed contexts; not independent or human evidence."
                  if visual_support else
                  "Same-family observer and entailment judge plus unreviewed generated contexts; "
                  "visual invalidity is approximated from the degenerate joint screen.")
    report = {"status": "provisional_not_evidence",
              "warning": limitation,
              "images": len(selected_ids), "entailment_counts": dict(Counter(row["label"] for row in rows)),
              "visually_invalid_candidates": len(visually_invalid),
              "visual_support_counts": dict(Counter(visual_support.values())) if visual_support else None,
              "e1_natural_new_correct_rate": methods,
              "e1_image_bootstrap": method_intervals,
              "oracle_minus_coordinate": methods["recognition_oracle"] - methods["coordinate_planner"],
              "oracle_minus_montage": methods["recognition_oracle"] - methods["montage_planner"],
              "oracle_minus_coordinate_bootstrap": oracle_coordinate_ci,
              "oracle_minus_montage_bootstrap": oracle_montage_ci,
              "e2_oracle_data_screen": {"caption_necessity_gap_mean": mean(necessity),
                                        "natural_to_saturated_switch_rate": mean(switches),
                                        "saturated_stop_oracle_rate": mean(saturated_stops),
                                        "selected_action_redundancy_loss": mean(redundancy_losses),
                                        "image_bootstrap": {
                                            "caption_necessity_gap": bootstrap_cluster(
                                                necessity, selected_ids, samples=10000, seed=2271),
                                            "switch_rate": bootstrap_cluster(
                                                switches, selected_ids, samples=10000, seed=2271),
                                            "saturated_stop_rate": bootstrap_cluster(
                                                saturated_stops, selected_ids, samples=10000, seed=2271),
                                            "redundancy_loss": bootstrap_cluster(
                                                redundancy_losses, selected_ids[:len(redundancy_losses)],
                                                samples=10000, seed=2271),
                                        }},
              "per_image_method_rows": per_image}
    (args.output / "provisional_report.json").write_text(json.dumps(report, indent=2,
                                                                      sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "per_image_method_rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
