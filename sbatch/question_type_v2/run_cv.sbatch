#!/bin/bash
#SBATCH --job-name=qtv2_cv
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/qtv2_cv_%A_%a.out
#SBATCH --error=logs/qtv2_cv_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1

# idx: name | model | context | max_len | lr | weighting
CFGS=(
 "E1_base_nocontext_cw|xlm-roberta-base|none|128|2e-5|--class_weighted"
 "E2_base_nocontext_cwsqrt|xlm-roberta-base|none|128|2e-5|--class_weighted --weight_power 0.5"
 "E3_base_context_cw|xlm-roberta-base|prev|192|2e-5|--class_weighted"
 "E4_large_nocontext_cw|xlm-roberta-large|none|128|1e-5|--class_weighted"
 "E5_large_context_cw|xlm-roberta-large|prev|192|1e-5|--class_weighted"
 "E6_base_nocontext_unweighted|xlm-roberta-base|none|128|2e-5|"
)
IFS='|' read -r NAME MODEL CTX ML LR W <<< "${CFGS[$SLURM_ARRAY_TASK_ID]}"
python src/question_type_v2/train_v2.py --mode cv \
  --train_csv results/question_type_v2/train_dev.csv \
  --out_dir results/question_type_v2/experiments/$NAME \
  --model $MODEL --context $CTX --max_length $ML --lr $LR --epochs 5 $W
