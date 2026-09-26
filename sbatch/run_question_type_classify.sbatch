#!/bin/bash
#SBATCH --job-name=qtype_classify
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/qtype_classify_%j.out
#SBATCH --error=logs/qtype_classify_%j.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311

export HF_HUB_OFFLINE=1

python src/question_type_analysis.py \
    --episodes results/features/balanced_200_episodes.csv \
    --transcript_dir data/outputs/whisperx \
    --model_path models/xlmr_question_type \
    --out_dir results \
    --device cuda \
    --batch_size 64

echo "Job finished at $(date)"
