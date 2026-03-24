#!/bin/bash
#SBATCH --job-name=whisperx_diarize
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== SLURM JOB INFO ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "CWD:   $(pwd)"
echo "======================"

# Go to your repo root if needed:
# cd /path/to/your/repo

# Activate conda env (most HPC conda installs need this)
module purge || true

# If conda is available as a module, uncomment one of these:
# module load anaconda3
# module load miniconda3

# Robust conda activation (works for many clusters)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # fallback: try to initialize conda in this shell
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found. If you're using venv instead, tell me the activate path."
  exit 1
fi

conda activate lbtll-py311

python -V
which python
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Run
python src/diarizze_whisperx_to_txt_v2.py \
  --audio data/raw_audio/Sunita_Williams_full.wav \
  --outdir outputs \
  --model medium \
  --compute_type int8 \
  --language en \
  --min_speakers 2 \
  --max_speakers 4

echo "End: $(date)"

