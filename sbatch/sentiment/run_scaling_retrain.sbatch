#!/bin/bash
#SBATCH --job-name=sent_scale
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/sent_scale_%j.out
#SBATCH --error=logs/sent_scale_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
python src/sentiment/scaling_retrain.py
echo "Job finished at $(date)"
