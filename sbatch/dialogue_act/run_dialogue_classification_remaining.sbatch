#!/bin/bash
#SBATCH --job-name=dialogue_act_218
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=10:30:00
#SBATCH --output=logs/dialogue_act_218_%j.out
#SBATCH --error=logs/dialogue_act_218_%j.err

# Dialogue act classification for the 218 episodes not in
# balanced_200_episodes.csv (already classified by run_dialogue_classification.sbatch),
# with the same script and settings. Output goes to its own folder so the
# 200-episode results are kept; the two are combined afterwards.
#
# Usage:
#   sbatch sbatch/dialogue_act/run_dialogue_classification_remaining.sbatch

cd /home/sr5868/listening-between-the-lines
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

echo "Started at  : $(date)"

python src/dialogue_act/dialogue_act_classification.py \
    --input_glob "data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json" \
    --balanced_csv results/dialogue_acts/remaining_218_episodes.csv \
    --out_dir results/dialogue_acts/remaining_218/ \
    --max_episodes 218 \
    --use_cuda \
    --batch_size 32

EXIT_CODE=$?
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
exit $EXIT_CODE
