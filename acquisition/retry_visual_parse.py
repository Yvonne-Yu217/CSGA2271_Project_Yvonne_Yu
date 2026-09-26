"""Retry only unparsed independent visual judgments with coded labels."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from acquisition.observe import atomic_jsonl, crop_candidate, read_jsonl
from acquisition.screen_independent_visual import DEFAULT_MODEL, DEFAULT_REVISION


RETRY_PROMPT = """Candidate observation: {observation}
Judge only against the supplied crop. Choose exactly one code and output only that code:
A = every concrete claim is directly visible (SUPPORTED)
B = some concrete claim is visible but another is unsupported (PARTIAL)
C = no concrete claim is supported or it relies on inference (INVALID)
D = it states no recognizable concrete fact (UNINFORMATIVE)
Answer only A, B, C, or D."""
CODES = {"A": "SUPPORTED", "B": "PARTIAL", "C": "INVALID", "D": "UNINFORMATIVE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--observer-output", type=Path, required=True)
    parser.add_argument("--screen-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows_path = args.screen_output / "visual_support.jsonl"
    rows = {row["candidate_id"]: row for row in read_jsonl(rows_path)}
    pending_ids = sorted(key for key, row in rows.items() if row.get("status") != "ok")
    if not pending_ids:
        print("no visual parse failures to retry")
        return
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    candidates = {row["candidate_id"]: row
                  for row in read_jsonl(args.staging / "automatic_candidates.jsonl")}
    observations = {row["candidate_id"]: row
                    for row in read_jsonl(args.observer_output / "observations.jsonl")}
    processor = AutoProcessor.from_pretrained(
        DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True,
        size={"longest_edge": 768})
    processor.tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    started = time.time()
    for offset in range(0, len(pending_ids), args.batch_size):
        chunk_ids = pending_ids[offset:offset + args.batch_size]
        crops, texts = [], []
        for candidate_id in chunk_ids:
            candidate = candidates[candidate_id]
            crops.append(crop_candidate(args.staging, images[candidate["staging_image_id"]],
                                        candidate))
            prompt = RETRY_PROMPT.format(
                observation=observations[candidate_id]["observed_text"])
            conversation = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(processor.apply_chat_template(conversation,
                                                       add_generation_prompt=True))
        inputs = processor(text=texts, images=crops, padding=True,
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=4,
                                       do_sample=False, use_cache=True)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        for candidate_id, raw in zip(chunk_ids, decoded):
            code = raw.strip().upper().strip(".:- ")
            rows[candidate_id].update({
                "raw_output_initial": rows[candidate_id].get("raw_output"),
                "raw_output": raw.strip(), "retry_code": code,
                "retry_prompt_sha256": hashlib.sha256(RETRY_PROMPT.encode()).hexdigest(),
                "label": CODES.get(code), "status": "ok" if code in CODES else "parse_failed",
            })
        for crop in crops:
            crop.close()
        atomic_jsonl(rows_path, [rows[key] for key in sorted(rows)])
    failures = [row for row in rows.values() if row.get("status") != "ok"]
    report = {
        "status": "complete" if not failures else "fail",
        "scope": "strict coded retry of parse failures only; not human gold",
        "retried": len(pending_ids), "remaining_failures": len(failures),
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "prompt_sha256": hashlib.sha256(RETRY_PROMPT.encode()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "counts": dict(Counter(row.get("label") for row in rows.values())),
    }
    (args.screen_output / "retry_runtime.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.screen_output / "audit.json").write_text(json.dumps({
        "status": report["status"],
        "scope": "independent VLM sensitivity only; not human gold",
        "counts": report["counts"], "rows": len(rows), "expected": len(rows),
        "parse_failures": [[row.get("raw_output"), 1] for row in failures],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"unparsed coded retries: {len(failures)}")


if __name__ == "__main__":
    main()
