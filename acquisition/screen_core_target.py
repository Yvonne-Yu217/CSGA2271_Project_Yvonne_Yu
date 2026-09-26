"""Classify provisional new facts by whether they concern an already-mentioned entity."""
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


LABELS = ("MENTIONED_ENTITY_DETAIL", "NEW_ENTITY", "OTHER", "NOT_NEW")
PROMPT = """Existing caption: {caption}
New candidate observation: {observation}
Classify what the additional information primarily contributes:
MENTIONED_ENTITY_DETAIL = an attribute, action, state, or relation involving an entity already
explicitly mentioned in the caption; NEW_ENTITY = primarily introduces an unmentioned entity;
OTHER = other new scene/background information; NOT_NEW = no additional fact.
If any new mentioned-entity detail is present, prefer MENTIONED_ENTITY_DETAIL.
Reply with exactly one label."""

STRICT_PROMPT = """Existing caption: {caption}
Candidate observation: {observation}
Assign exactly one label. MENTIONED_ENTITY_DETAIL only when the observation adds a visible
attribute, action, state, or relation about the same specific entity instance explicitly named in
the caption. A shared category alone (for example, another person) is not enough. NEW_ENTITY means
the added fact is mainly about a distinct entity absent from the caption. OTHER means new
background/global information or ambiguous instance linkage. NOT_NEW means no added fact.
Be conservative: if same-instance linkage is uncertain, use OTHER. Reply with only the label."""


def parse_label(text):
    value = text.strip().upper().strip(".:-").replace(" ", "_")
    return value if value in LABELS else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--visual-support-output", type=Path, required=True)
    parser.add_argument("--natural-e1-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt-variant", choices=("default", "strict"), default="default")
    parser.add_argument("--novel-entailment-label", default="NOT_ENTAILED",
                        help="NLI label treated as novel (for example NOT_ENTAILED or NEUTRAL)")
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
    entailments = {row["key"]: row for row in read_jsonl(
        args.natural_e1_output / "entailments.jsonl")}
    work = []
    for key, row in entailments.items():
        context = contexts[row["context_id"]]
        observation = observations[row["candidate_id"]]["observed_text"]
        eligible = (row["label"] == args.novel_entailment_label and
                    visual[row["candidate_id"]] == "SUPPORTED")
        work.append({"key": key, "context_id": row["context_id"],
                     "staging_image_id": row["staging_image_id"],
                     "candidate_id": row["candidate_id"], "caption": context["initial_caption"],
                     "observation": observation, "eligible": eligible})
    sources = [args.staging / "natural_contexts.jsonl",
               args.observer_output / "observations.jsonl",
               args.visual_support_output / "visual_support.jsonl",
               args.natural_e1_output / "entailments.jsonl"]
    prompt_template = STRICT_PROMPT if args.prompt_variant == "strict" else PROMPT
    payload = {"files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "model": args.model, "revision": args.revision, "prompt": prompt_template,
               "batch_size": args.batch_size,
               "novel_entailment_label": args.novel_entailment_label,
               "implementation": "provisional-core-target-v2-configurable-nli-label"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, metadata_path = args.output / "complement_types.jsonl", args.output / "runtime.json"
    existing = {row["key"]: row for row in read_jsonl(rows_path)}
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if old_metadata and old_metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different core-target screen")
    pending = []
    for row in work:
        if row["key"] in existing:
            continue
        if not row["eligible"]:
            existing[row["key"]] = {"key": row["key"], "context_id": row["context_id"],
                                    "staging_image_id": row["staging_image_id"],
                                    "candidate_id": row["candidate_id"], "label": "NOT_NEW",
                                    "raw_output": "deterministic:not-eligible", "status": "ok"}
        else:
            pending.append(row)
    metadata = {"scope": "provisional same-model semantic type screen; not human gold",
                "fingerprint": fingerprint, "model": args.model, "revision": args.revision,
                "prompt_variant": args.prompt_variant,
                "novel_entailment_label": args.novel_entailment_label,
                "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
                "expected": len(work), "completed": len(existing),
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
                prompt = prompt_template.format(caption=row["caption"], observation=row["observation"])
                conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                texts.append(processor.apply_chat_template(conversation, tokenize=False,
                                                           add_generation_prompt=True))
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
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
            print(f"core target {len(existing)}/{len(work)}", flush=True)
        metadata.update({"status": "complete", "completed": len(existing),
                         "elapsed_seconds": time.time() - started,
                         "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                         "gpu_name": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = list(existing.values())
    if len(rows) != len(work) or any(row["status"] != "ok" for row in rows):
        failures = Counter(row.get("raw_output") for row in rows if row.get("status") != "ok")
        raise RuntimeError(f"core-target cache incomplete/unparsed: {failures.most_common(5)}")
    type_lookup = {(row["context_id"], row["candidate_id"]): row["label"] for row in rows}
    natural_report = json.loads((args.natural_e1_output / "provisional_report.json").read_text())
    base_rows = natural_report["context_rows"]
    selected = {(row["context_id"], row["method"]): row["candidate_id"] for row in base_rows
                if row["candidate_id"] is not None}
    methods = ("montage_planner", "coordinate_planner", "clip_inverse",
               "clip_inverse_cost_matched", "largest_area")
    result_rows = []
    for context_id, context in contexts.items():
        image_id = context["staging_image_id"]
        nonstops = [row for row in candidates_by_image[image_id] if row["kind"] != "stop"]
        values = {row["candidate_id"]: int(
            type_lookup[(context_id, row["candidate_id"])] == "MENTIONED_ENTITY_DETAIL")
                  for row in nonstops}
        for method in methods:
            candidate_id = selected[(context_id, method)]
            result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                                "method": method, "candidate_id": candidate_id,
                                "core_target_success": values.get(candidate_id, 0)})
        oracle_id = min(values, key=lambda cid: (-values[cid], candidates[cid]["pixel_cost"], cid))
        result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                            "method": "core_target_oracle", "candidate_id": oracle_id,
                            "core_target_success": values[oracle_id]})
        result_rows.append({"context_id": context_id, "staging_image_id": image_id,
                            "method": "random_expected", "candidate_id": None,
                            "core_target_success": sum(values.values()) / len(values)})
    summaries = {}
    for method in sorted({row["method"] for row in result_rows}):
        subset = [row for row in result_rows if row["method"] == method]
        summaries[method] = bootstrap_cluster(
            [row["core_target_success"] for row in subset],
            [row["staging_image_id"] for row in subset], samples=10000, seed=2271)
    by_key = {(row["context_id"], row["method"]): row["core_target_success"]
              for row in result_rows}
    context_ids = sorted(contexts)
    clusters = [contexts[context_id]["staging_image_id"] for context_id in context_ids]
    gaps = {method: paired_bootstrap(
        [by_key[(context_id, "core_target_oracle")] for context_id in context_ids],
        [by_key[(context_id, method)] for context_id in context_ids], clusters,
        samples=10000, seed=2271) for method in methods}
    report = {"status": "provisional_not_evidence",
              "warning": ("Automated semantic classifier over generated observer outputs; "
                          "no independent human fact typing."),
              "images": len({row["staging_image_id"] for row in contexts.values()}),
              "contexts": len(contexts), "type_counts": dict(Counter(row["label"] for row in rows)),
              "core_target_success": summaries, "oracle_minus_baseline": gaps,
              "context_rows": result_rows}
    (args.output / "provisional_report.json").write_text(json.dumps(report, indent=2,
                                                                      sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "images", "contexts", "type_counts",
                                                   "core_target_success", "oracle_minus_baseline")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
