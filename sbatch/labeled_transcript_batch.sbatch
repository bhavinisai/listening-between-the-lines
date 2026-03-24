#!/bin/bash
#SBATCH --job-name=label_all_episodes
#SBATCH --partition=tier3
#SBATCH --output=logs/label_all_%j.log
#SBATCH --error=logs/label_all_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Create logs directory first
mkdir -p ~/listening-between-the-lines/logs

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/listening-between-the-lines

source ~/miniconda3/bin/activate
conda activate lbtll-py311

INPUT_DIR="outputs"
PROCESSED=0
FAILED=0

echo "Processing all episodes in $INPUT_DIR"
echo "----------------------------------------"

for json_file in "$INPUT_DIR"/Episode*.whisperx.json; do
    if [ -f "$json_file" ]; then
        echo ""
        echo "Processing: $(basename $json_file)"

        if python src/labeled_transcript_v3.py --json "$json_file" --interview-style; then
            echo "✓ Success: $(basename $json_file)"
            ((PROCESSED++))
        else
            echo "✗ Failed: $(basename $json_file)"
            ((FAILED++))
        fi
    fi
done

echo ""
echo "========================================="
echo "Done! Processed: $PROCESSED | Failed: $FAILED"
echo "Job finished at: $(date)"
