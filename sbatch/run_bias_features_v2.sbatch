#!/bin/bash
#SBATCH --job-name=bias_feat_v2
#SBATCH --output=logs/bias_features_v2_%j.out
#SBATCH --error=logs/bias_features_v2_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=tier3

# CPU only: Hyland lexicon matching + ConvoKit PolitenessStrategies (spaCy parse).
#
# Usage:
#   sbatch sbatch/run_bias_features_v2.sbatch

cd /home/sr5868/listening-between-the-lines

source ~/miniconda3/etc/profile.d/conda.sh
conda activate convokit

echo "Started at  : $(date)"

python src/compute_bias_features_v2.py \
    --input_glob "data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json" \
    --out_csv results/features/speaker_features_v2.csv \
    --terms_csv results/features/speaker_feature_terms_v2.csv \
    --audit_csv results/features/audit_sample_v2.csv \
    --audit_n 50

EXIT_CODE=$?
echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"
exit $EXIT_CODE
