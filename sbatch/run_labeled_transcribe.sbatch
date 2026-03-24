#!/bin/bash
#SBATCH --job-name=identify_host
#SBATCH --partition=tier3
#SBATCH --output=logs/identify_host_%j.log
#SBATCH --error=logs/identify_host_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Log job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to project directory
cd ~/listening-between-the-lines

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate lbtll-py311

# Run the identification script
python src/diarizze_whisperx_to_txt_v3.py \
  --json outputs/whisperx_output/Sunita_Williams_full.whisperx.json \
  --show_sponsor_segments

# Log completion
echo "Job finished at: $(date)"
