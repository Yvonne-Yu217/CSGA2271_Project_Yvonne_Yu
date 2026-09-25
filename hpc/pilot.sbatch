#!/bin/bash
#SBATCH --job-name=visual-complementarity-pilot
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${PROJECT_DATA_DIR:?Set PROJECT_DATA_DIR to the prepared pilot data directory}"
: "${PROJECT_RESULTS_DIR:?Set PROJECT_RESULTS_DIR to persistent experiment storage}"
test -s "$PROJECT_DATA_DIR/manifest.json"
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
nvidia-smi
python3 -u pilot/run.py --device cuda --batch-size 32 \
  --data-dir "$PROJECT_DATA_DIR" \
  --output-dir "$PROJECT_RESULTS_DIR/pilot-${SLURM_JOB_ID}"
