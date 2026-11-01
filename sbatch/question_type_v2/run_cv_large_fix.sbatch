#!/bin/bash
# Re-run of E4/E5 only (tasks 3-4 of job 21979202 failed: model id must be the
# cached repo id FacebookAI/xlm-roberta-large when HF_HUB_OFFLINE=1).
# Same settings as run_cv.sbatch; E1-E3, E6 are NOT rerun.
#SBATCH --job-name=qtv2_cv_large
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/qtv2_cv_large_%A_%a.out
#SBATCH --error=logs/qtv2_cv_large_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1

CFGS=(
 "E4_large_nocontext_cw|FacebookAI/xlm-roberta-large|none|128|1e-5|--class_weighted"
 "E5_large_context_cw|FacebookAI/xlm-roberta-large|prev|192|1e-5|--class_weighted"
)
IFS='|' read -r NAME MODEL CTX ML LR W <<< "${CFGS[$SLURM_ARRAY_TASK_ID]}"
python src/question_type_v2/train_v2.py --mode cv \
  --train_csv results/question_type_v2/train_dev.csv \
  --out_dir results/question_type_v2/experiments/$NAME \
  --model $MODEL --context $CTX --max_length $ML --lr $LR --epochs 5 $W
