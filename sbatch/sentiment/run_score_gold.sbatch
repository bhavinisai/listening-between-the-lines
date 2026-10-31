#!/bin/bash
#SBATCH --job-name=sent_gold
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/sent_gold_%j.out
#SBATCH --error=logs/sent_gold_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
python src/sentiment/score_models.py --input results/sentiment/gold_sheet_labeled.csv \
    --out results/sentiment/gold_model_probs.csv
