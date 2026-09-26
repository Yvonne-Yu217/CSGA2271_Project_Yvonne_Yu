"""Cache CLIP crop/context scores for E1 simple baselines.

This is model-visible preprocessing only.  It never reads ``gold/`` and cannot
produce an E1 conclusion without separately adjudicated action outcomes.
"""
import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(paths, model, revision, batch_size):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(model.encode())
    digest.update(revision.encode())
    digest.update(str(batch_size).encode())
    digest.update(torch.__version__.encode())
    digest.update(__import__("transformers").__version__.encode())
    digest.update(b"clip-baseline-v2-ngram4")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--revision", default="57c216476eefef5ab752ec549e440a49ae4ae5f3")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.staging / "staging_images.jsonl"
    context_path = args.staging / "natural_contexts.jsonl"
    candidate_path = args.staging / "automatic_candidates.jsonl"
    run_fingerprint = fingerprint((image_path, context_path, candidate_path), args.model,
                                  args.revision, args.batch_size)
    scores_path = args.output / "scores.jsonl"
    runtime_path = args.output / "runtime.json"
    context_manifest = read_jsonl(context_path)
    candidate_manifest = [row for row in read_jsonl(candidate_path) if row["kind"] != "stop"]
    candidate_counts = {}
    for row in candidate_manifest:
        candidate_counts[row["staging_image_id"]] = candidate_counts.get(row["staging_image_id"], 0) + 1
    expected_score_rows = sum(candidate_counts.get(row["staging_image_id"], 0)
                              for row in context_manifest)
    if scores_path.is_file() and runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text())
        score_bytes = scores_path.read_bytes()
        valid_rows = sum(bool(line.strip()) for line in score_bytes.splitlines())
        if (runtime.get("fingerprint") == run_fingerprint and
                runtime.get("scores_sha256") == hashlib.sha256(score_bytes).hexdigest() and
                runtime.get("score_rows") == valid_rows == expected_score_rows):
            print(f"cache hit: {scores_path}")
            return

    images = {row["staging_image_id"]: row for row in read_jsonl(image_path)}
    contexts = context_manifest
    candidates = candidate_manifest
    model = AutoModel.from_pretrained(args.model, revision=args.revision,
                                      local_files_only=not args.allow_model_download).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                               local_files_only=not args.allow_model_download,
                                               use_fast=False)
    started = time.time()

    def encode_text(rows):
        output = []
        for offset in range(0, len(rows), args.batch_size):
            chunk = rows[offset:offset + args.batch_size]
            batch = processor(text=chunk, return_tensors="pt", padding=True, truncation=True,
                              max_length=77).to(device)
            with torch.inference_mode():
                features = model.get_text_features(**batch)
                features = F.normalize(features.float(), dim=-1)
            output.append(features.cpu().numpy())
            print(f"text {min(offset + len(chunk), len(rows))}/{len(rows)}", flush=True)
        return np.concatenate(output)

    def load_candidate(candidate):
        image_row = images[candidate["staging_image_id"]]
        with Image.open(args.staging / image_row["image_path"]) as source:
            source = source.convert("RGB")
            crops = [source.crop(box) for box in candidate["boxes"]]
        return crops

    candidate_features = []
    for offset in range(0, len(candidates), args.batch_size):
        chunk = candidates[offset:offset + args.batch_size]
        flat, spans = [], []
        for candidate in chunk:
            crops = load_candidate(candidate)
            spans.append((len(flat), len(flat) + len(crops)))
            flat.extend(crops)
        batch = processor(images=flat, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = F.normalize(model.get_image_features(**batch).float(), dim=-1)
        for start, end in spans:
            candidate_features.append(F.normalize(features[start:end].mean(0), dim=-1).cpu().numpy())
        print(f"crop {min(offset + len(chunk), len(candidates))}/{len(candidates)}", flush=True)
    candidate_features = np.stack(candidate_features)
    text_features = encode_text([row["initial_caption"] for row in contexts])
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "is",
                 "are", "to", "by", "for", "from", "this", "that"}
    spans_by_context = []
    for context in contexts:
        words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", context["initial_caption"].lower())
        spans = sorted({" ".join(words[start:start + length])
                        for length in range(1, 5)
                        for start in range(len(words) - length + 1)
                        if any(word not in stopwords for word in words[start:start + length])})
        spans_by_context.append(spans or [context["initial_caption"]])
    unique_spans = sorted({span for spans in spans_by_context for span in spans})
    span_index = {span: index for index, span in enumerate(unique_spans)}
    span_features = encode_text(["a photo of " + span for span in unique_spans])

    candidates_by_image = {}
    for index, candidate in enumerate(candidates):
        candidates_by_image.setdefault(candidate["staging_image_id"], []).append((index, candidate))
    rows = []
    for text_index, context in enumerate(contexts):
        for candidate_index, candidate in candidates_by_image[context["staging_image_id"]]:
            cosine = float(candidate_features[candidate_index] @ text_features[text_index])
            candidate_spans = span_features[[span_index[span] for span in spans_by_context[text_index]]]
            max_ngram_cosine = float((candidate_spans @ candidate_features[candidate_index]).max())
            rows.append({
                "context_id": context["context_id"], "candidate_id": candidate["candidate_id"],
                "staging_image_id": context["staging_image_id"], "cosine_similarity": cosine,
                "inverse_cosine_score": 0.5 * (1.0 - cosine),
                "max_ngram_cosine_similarity": max_ngram_cosine,
                "max_ngram_inverse_score": 0.5 * (1.0 - max_ngram_cosine),
            })
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = scores_path.with_suffix(".jsonl.partial")
    temporary.write_text(rendered)
    temporary.replace(scores_path)
    runtime = {
        "scope": "model-visible baseline cache only; no E1/E2 outcome claim",
        "fingerprint": run_fingerprint, "model": args.model, "revision": args.revision,
        "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "images": len(images), "contexts": len(contexts), "candidates_without_stop": len(candidates),
        "ngram_spans": len(unique_spans), "score_rows": len(rows),
        "seconds": time.time() - started, "batch_size": args.batch_size,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "scores_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
