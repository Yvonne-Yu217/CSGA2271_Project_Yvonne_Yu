"""Automatic caption-derived Grounding DINO proposals for the R1 geometry arm."""
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from acquisition.observe import atomic_jsonl, read_jsonl


MODEL = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
NMS_IOU = 0.70
MAX_DETECTIONS = 6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    images = {row["staging_image_id"]: row
              for row in read_jsonl(args.staging / "staging_images.jsonl")}
    contexts_by_image = defaultdict(list)
    for row in read_jsonl(args.staging / "natural_contexts.jsonl"):
        contexts_by_image[row["staging_image_id"]].append(row)
    source_paths = [args.staging / "staging_images.jsonl",
                    args.staging / "natural_contexts.jsonl"]
    payload = {
        "implementation": "grounded-regions-r1-v2-fp32",
        "files": {str(path): digest(path) for path in source_paths},
        "model": args.model, "revision": args.revision,
        "model_dtype": "float32",
        "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD,
        "nms_iou": NMS_IOU, "max_detections": MAX_DETECTIONS,
        "query_rule": "all five natural captions joined in sorted context-id order",
        "batch_size": args.batch_size,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path, runtime_path = args.output / "grounded_regions.jsonl", args.output / "runtime.json"
    old = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if old and old.get("fingerprint") != fingerprint:
        raise RuntimeError("output belongs to a different grounding fingerprint")
    existing = {row["staging_image_id"]: row for row in read_jsonl(rows_path)}
    pending_ids = [image_id for image_id in sorted(images) if image_id not in existing]
    runtime = {
        **payload, "fingerprint": fingerprint,
        "scope": "model-visible automatic proposals; no gold regions",
        "expected_images": len(images), "completed_images": len(existing),
        "status": "running" if pending_ids else "complete",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
    if not pending_ids:
        print("grounding cache complete")
        return
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision,
        local_files_only=not args.allow_model_download)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model, revision=args.revision,
        local_files_only=not args.allow_model_download,
        dtype=torch.float32).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for offset in range(0, len(pending_ids), args.batch_size):
        chunk_ids = pending_ids[offset:offset + args.batch_size]
        batch_images, queries, context_ids = [], [], []
        for image_id in chunk_ids:
            image_row = images[image_id]
            path = args.staging / image_row["image_path"]
            if digest(path) != image_row["image_sha256"]:
                raise RuntimeError(f"image hash mismatch: {image_id}")
            batch_images.append(Image.open(path).convert("RGB"))
            context_rows = sorted(contexts_by_image[image_id], key=lambda row: row["context_id"])
            context_ids.append([row["context_id"] for row in context_rows])
            queries.append(" . ".join(row["initial_caption"].rstrip(" .")
                                       for row in context_rows) + " .")
        inputs = processor(images=batch_images, text=queries, padding=True,
                           return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = [(image.height, image.width) for image in batch_images]
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD, target_sizes=target_sizes)
        for image_id, query, ids, result in zip(chunk_ids, queries, context_ids, results):
            image_row = images[image_id]
            boxes = result["boxes"].float().cpu()
            scores = result["scores"].float().cpu()
            labels = result.get("text_labels", result.get("labels", []))
            keep = nms(boxes, scores, NMS_IOU)[:MAX_DETECTIONS].tolist() if len(boxes) else []
            detections = []
            for index in keep:
                x1, y1, x2, y2 = boxes[index].tolist()
                box = [max(0, int(round(x1))), max(0, int(round(y1))),
                       min(image_row["width"], int(round(x2))),
                       min(image_row["height"], int(round(y2)))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                label = labels[index] if index < len(labels) else ""
                if not isinstance(label, str):
                    label = str(label)
                detections.append({"box": box, "score": float(scores[index]),
                                   "text_label": label})
            existing[image_id] = {
                "staging_image_id": image_id, "image_sha256": image_row["image_sha256"],
                "query": query, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "query_context_ids": ids, "raw_detection_count": len(boxes),
                "detections": detections, "status": "ok" if detections else "no_detection",
            }
        for image in batch_images:
            image.close()
        atomic_jsonl(rows_path, [existing[key] for key in sorted(existing)])
        runtime.update({
            "completed_images": len(existing), "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        })
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        print(f"grounding {len(existing)}/{len(images)}", flush=True)
    runtime.update({
        "status": "complete", "completed_images": len(existing),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "gpu_name": torch.cuda.get_device_name(0),
        "images_without_detection": sum(row["status"] == "no_detection"
                                         for row in existing.values()),
    })
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
