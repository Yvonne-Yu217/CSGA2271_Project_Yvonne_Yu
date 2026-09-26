"""Sample useful-process and GPU progress without creating accelerator load."""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


QUERY = ("index,name,utilization.gpu,memory.used,memory.total,power.draw,"
         "clocks.current.sm,clocks_throttle_reasons.active")


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gpu_rows():
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    return list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = QUERY.split(",")
    started = time.monotonic()
    with args.output.open("a", encoding="utf-8") as handle:
        while process_alive(args.pid):
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            progress = {}
            if args.progress_file and args.progress_file.exists():
                stat = args.progress_file.stat()
                progress = {"progress_bytes": stat.st_size,
                            "progress_mtime_unix": stat.st_mtime}
            try:
                rows = gpu_rows()
                error = None
            except (OSError, subprocess.SubprocessError) as exc:
                rows, error = [], str(exc)
            record = {
                "timestamp_utc": now, "job_id": os.getenv("SLURM_JOB_ID"),
                "run_id": args.run_id, "pid": args.pid,
                "elapsed_seconds": time.monotonic() - started,
                "gpus": [dict(zip(fields, row)) for row in rows], **progress,
            }
            if error:
                record["monitor_error"] = error
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
