#!/bin/bash
#SBATCH -p sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=whisperx
#SBATCH --output=logs/whisperx_%A_%a.out
#SBATCH --error=logs/whisperx_%A_%a.err
#SBATCH --array=1-299

set -euo pipefail

mkdir -p logs
mkdir -p data/outputs

cd /home/sr5868/listening-between-the-lines

source /home/sr5868/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

AUDIO_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" audio_files.txt)

if [ -z "${AUDIO_FILE}" ]; then
  echo "No audio file found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

BASENAME=$(basename "$AUDIO_FILE")
STEM="${BASENAME%.*}"

echo "Processing: $AUDIO_FILE"
echo "Output stem: $STEM"
echo "Running on host: $(hostname)"

python src/diarizze_whisperx_gpu.py "$AUDIO_FILE" \
  --output_dir data/outputs/whisperx \
  --model large-v2 \
  --batch_size 16 \
  --hf_token "$HF_TOKEN" \
  --language en

echo "Done: $AUDIO_FILE"
