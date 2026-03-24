#!/bin/bash
#SBATCH --job-name=detect_gender
#SBATCH --partition=tier3
#SBATCH --output=logs/detect_gender_%j.log
#SBATCH --error=logs/detect_gender_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Log job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to project directory
cd ~/listening-between-the-lines

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate lbtll-py311

# Run gender detection
python src/detect_gender.py \
  --audio data/raw_audio/Sunita_Williams_full.wav \
  --json outputs/whisperx_output/Sunita_Williams_full.whisperx.labeled.json

# Log completion
echo "Job finished at: $(date)"
