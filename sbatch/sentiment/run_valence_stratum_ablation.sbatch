#!/bin/bash
#SBATCH --job-name=val_strat
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --output=logs/val_strat_%j.out
#SBATCH --error=logs/val_strat_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
python src/sentiment/valence_stratum_ablation.py
echo "Job finished at $(date)"
