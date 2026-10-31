#!/bin/bash
#SBATCH --job-name=sent_pipe
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=logs/sent_pipe_%j.out
#SBATCH --error=logs/sent_pipe_%j.err

# 1) episode-grouped CV sweep of valence fine-tuning configs on the gold set
# 2) pick the config with the best CV macro-F1, train it on all gold rows
# 3) score every turn (valence, zero-shot valence, emotion/arousal)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1
export PYTHONPATH=src/sentiment

R=results/sentiment/ft_cv
mkdir -p $R
for BASE in cardiff_xlmr cardiff_xlmr_multi; do
  for LR in 1e-5 3e-5; do
    for WP in 1.0 0.5; do
      TAG=${BASE}_lr${LR}_wp${WP}
      [ -f $R/$TAG.csv ] && continue
      echo "=== $TAG"
      python src/sentiment/finetune_valence.py --base $BASE --lr $LR --weight_pow $WP \
          --epochs 6 --seeds 2 --cv_out $R/$TAG.csv
    done
  done
done

BEST=$(python src/sentiment/select_best_cv.py $R)
echo "best config: $BEST"
BASE=$(echo $BEST | sed 's/_lr.*//'); LR=$(echo $BEST | sed 's/.*_lr\([^_]*\)_wp.*/\1/'); WP=$(echo $BEST | sed 's/.*_wp//')
python src/sentiment/finetune_valence.py --base $BASE --lr $LR --weight_pow $WP --epochs 6 \
    --final_dir models/valence_ft_final
python src/sentiment/score_corpus.py --valence_model models/valence_ft_final
echo "Job finished at $(date)"
