#!/bin/bash
#SBATCH --job-name=frozen-proposal-followup
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
SOURCE="${SOURCE:?Set SOURCE to the existing deletion run}"
OUT="${OUT:?Set OUT to the CPU-preflight output directory}"
DATA_DIR="${PROJECT_DATA_DIR:-$SLURM_SUBMIT_DIR/pilot/data}"
MODEL="${MODEL:-openai/clip-vit-base-patch16}"
test -s "$OUT/preflight.json"
test -x "$PYTHON"
# This timeout is a ceiling within the remaining phase budget, never new quota.
LIMIT_SECONDS="${LIMIT_SECONDS:-13800}"
if ! [[ "$LIMIT_SECONDS" =~ ^[0-9]+$ ]] || (( LIMIT_SECONDS < 1 || LIMIT_SECONDS > 13800 )); then
  echo 'LIMIT_SECONDS must be 1..13800 and within the reconciled remaining budget' >&2
  exit 2
fi
RUN_PID=''
MONITOR_PID=''
cleanup() {
  if [[ -n "$RUN_PID" ]]; then kill "$RUN_PID" 2>/dev/null || true; fi
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
timeout --signal=TERM --kill-after=30 "$LIMIT_SECONDS" "$PYTHON" -u pilot/next_round.py gpu \
  --source "$SOURCE" --data-dir "$DATA_DIR" --output "$OUT" --model "$MODEL" \
  --batch-size "${BATCH_SIZE:-64}" > "$OUT/gpu-${SLURM_JOB_ID}.log" 2>&1 &
RUN_PID=$!
(
  while kill -0 "$RUN_PID" 2>/dev/null; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader
    ps -u "$(id -u)" -o pid=,pcpu=,rss=,comm=
    squeue -h -j "$SLURM_JOB_ID" -o '%i %T %M %l' || true
    tail -n 2 "$OUT/gpu-${SLURM_JOB_ID}.log"
    sleep 5
  done
) > "$OUT/utilization-${SLURM_JOB_ID}.log" 2>&1 &
MONITOR_PID=$!
wait "$RUN_PID"
RUN_PID=''
echo 'GPU inference finished. Run the summary on a CPU allocation.'
