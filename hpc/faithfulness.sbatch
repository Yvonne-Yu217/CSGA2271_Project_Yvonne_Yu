#!/bin/bash
#SBATCH --job-name=visual-complementarity-faithfulness
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-${1:?Pass the formal tminus output directory as argument 1}}"
OUT="${PROJECT_RESULTS_DIR:-$SLURM_SUBMIT_DIR/pilot/results}/faithfulness-${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
("$PYTHON" -u pilot/faithfulness.py --device cuda --batch-size 32 --run-dir "$RUN_DIR" --output-dir "$OUT") &
RUN_PID=$!
(
  while kill -0 "$RUN_PID" 2>/dev/null; do
    printf '%s ' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader | tr '\n' ' '
    ps -p "$RUN_PID" -o %cpu=,rss= | awk '{print "cpu=" $1 "%,rss_kib=" $2}'
    sleep 2
  done
) > "$OUT/utilization.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
wait "$RUN_PID"
