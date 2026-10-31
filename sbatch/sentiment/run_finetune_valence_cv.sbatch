#!/bin/bash
#SBATCH --job-name=sent_ft_cv
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/sent_ft_cv_%j.out
#SBATCH --error=logs/sent_ft_cv_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1

R=results/sentiment/ft_cv
mkdir -p $R
for BASE in cardiff_xlmr cardiff_xlmr_multi; do
  for CTX in "" "--context"; do
    for LR in 1e-5 3e-5; do
      TAG=${BASE}_lr${LR}${CTX:+_ctx}
      echo "=== $TAG"
      python src/sentiment/finetune_valence.py --base $BASE --lr $LR $CTX --epochs 6 --seeds 3 \
          --cv_out $R/$TAG.csv
    done
  done
done
