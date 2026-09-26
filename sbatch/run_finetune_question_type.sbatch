#!/bin/bash
#SBATCH --job-name=xlmr_qtype
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/xlmr_qtype_%j.out
#SBATCH --error=logs/xlmr_qtype_%j.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311

# xlm-roberta-base was pre-downloaded on the login node (which has internet);
# compute nodes here don't, so force the HF cache to be used as-is.
export HF_HUB_OFFLINE=1

python src/finetune_question_type_xlmr.py \
    --labeled_csv results/question_type_seed_labeled.csv \
    --out_dir models/xlmr_question_type \
    --epochs 6 \
    --batch_size 16 \
    --class_weighted

echo "Job finished at $(date)"
