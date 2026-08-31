#!/bin/bash
#SBATCH --job-name=dialogue_act
#SBATCH --partition=sporc-gpu
#SBATCH --account=convflow
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=10:30:00
#SBATCH --output=logs/dialogue_act_%j.out
#SBATCH --error=logs/dialogue_act_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment (edit env name as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

# Run the classification script
python src/dialogue_act_classification.py \
    --input_glob "data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json" \
    --balanced_csv results/features/balanced_200_episodes.csv \
    --out_dir results/dialogue_acts/ \
    --max_episodes 200 \
    --use_cuda \
    --batch_size 32

echo "Job finished at $(date)"