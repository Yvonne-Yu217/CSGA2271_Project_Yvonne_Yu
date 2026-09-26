#!/bin/bash
#SBATCH --job-name=visual-complementarity-full-l4
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
DATA_DIR="${PROJECT_DATA_DIR:-$SLURM_SUBMIT_DIR/pilot/data}"
RESULTS_ROOT="${PROJECT_RESULTS_DIR:-$SLURM_SUBMIT_DIR/pilot/results}/formal-l4-${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
test -x "$PYTHON"
test -s "$DATA_DIR/manifest.json"
mkdir -p "$RESULTS_ROOT"

(
  while true; do
    printf '%s ' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader | tr '\n' ' '
    ps -u "$(id -u)" -o %cpu=,rss=,comm= | awk '{cpu+=$1; rss+=$2} END {print "cpu_sum=" cpu "%,rss_kib=" rss}'
    sleep 2
  done
) > "$RESULTS_ROOT/utilization.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

for field in tminus tmatched tnatural; do
  "$PYTHON" -u pilot/run.py --device cuda --batch-size 64 \
    --minus-field "$field" --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_ROOT/$field" 2>&1 | tee "$RESULTS_ROOT/$field.log"
done

"$PYTHON" -u pilot/baselines.py --device cuda --batch-size 32 \
  --minus-field tminus --data-dir "$DATA_DIR" \
  --output-dir "$RESULTS_ROOT/spatial-baselines" 2>&1 | tee "$RESULTS_ROOT/spatial-baselines.log"
