#!/bin/bash
#SBATCH --job-name=sent_llm2
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/sent_llm2_%j.out
#SBATCH --error=logs/sent_llm2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
python src/sentiment/llm_pipeline.py --scratch results/sentiment/llm_labels_passes --out_dir results/sentiment/llm_ft2
echo "Job finished at $(date)"
