#!/bin/bash
# v3-label CV: same host-grouped protocol and settings as E1/E3, trained on
# train_dev_v3.csv. One controlled comparison, no hyperparameter search.
#SBATCH --job-name=qtv3_cv
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/qtv3_cv_%A_%a.out
#SBATCH --error=logs/qtv3_cv_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1

CFGS=(
 "V3_E1_base_nocontext_cw|xlm-roberta-base|none|128|2e-5"
 "V3_E3_base_context_cw|xlm-roberta-base|prev|192|2e-5"
)
IFS='|' read -r NAME MODEL CTX ML LR <<< "${CFGS[$SLURM_ARRAY_TASK_ID]}"
python src/question_type_v2/train_v2.py --mode cv \
  --train_csv results/question_type_v2/v3/train_dev_v3.csv \
  --out_dir results/question_type_v2/experiments/$NAME \
  --model $MODEL --context $CTX --max_length $ML --lr $LR --epochs 5 --class_weighted
