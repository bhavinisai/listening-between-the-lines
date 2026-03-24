#!/bin/bash
#SBATCH --job-name=diarize_ep001
#SBATCH --partition=tier3
#SBATCH --output=logs/diarize_ep001_%j.log
#SBATCH --error=logs/diarize_ep001_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create logs directory first
mkdir -p ~/listening-between-the-lines/logs

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/listening-between-the-lines

source ~/miniconda3/bin/activate
conda activate lbtll-py311

# (Recommended) ensure HF token is available for pyannote
# export HF_TOKEN="hf_..."

echo "Running diarization + WhisperX on one episode..."
echo "----------------------------------------"

python src/diarizze_whisperx_to_txt_v5.py \
  --audio "data/raw_audio/episode_020_The NEXT 10 Years Of Indian Education System - Ronnie Screwvala & Raj Shamani on Figuring Out 107.wav" \
  --outdir "outputs/" \
  --model medium \
  --compute_type int8 \
  --language en \
  --min_speakers 2 \
  --max_speakers 4

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Success"
else
  echo "✗ Failed (exit code: $EXIT_CODE)"
fi
echo "Job finished at: $(date)"
exit $EXIT_CODE

