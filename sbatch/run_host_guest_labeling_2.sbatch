#!/bin/bash
#SBATCH --job-name=host_guest_label
#SBATCH --output=logs/host_guest_label_%A_%a.out
#SBATCH --error=logs/host_guest_label_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=tier3
#SBATCH --array=0-99   # Adjust to (number of episodes - 1)

# NOTE: No GPU needed — this script only calls an external API.
#
# Usage:
#   export GROQ_API_KEY="your_key_here"
#   sbatch run_host_guest_label.sh

cd /home/sr5868/listening-between-the-lines

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

# Validate API key was passed in from environment
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY is not set." >&2
    echo "       Run: export GROQ_API_KEY='your_key_here' before sbatch" >&2
    exit 1
fi

INPUT_DIR="data/outputs/whisperx"
OUTPUT_DIR="data/outputs/whisperx"

# Build array of input files
INPUT_FILES=(${INPUT_DIR}/*_whisperx_diarized.gender.json)
TOTAL=${#INPUT_FILES[@]}

# Guard: skip if this task ID is out of range (handles uneven file counts)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no file to process (only $TOTAL files found). Exiting."
    exit 0
fi

# Pick the file for this array task
INPUT_FILE="${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "${INPUT_FILE}" .json)

echo "Task ID     : $SLURM_ARRAY_TASK_ID"
echo "Input file  : ${INPUT_FILE}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Started at  : $(date)"

python src/labeled_transcript_v12.py \
    --input "${INPUT_FILE}" \
    --out_dir "${OUTPUT_DIR}" \
    --speaker_key speaker 
    

EXIT_CODE=$?

echo "Finished at : $(date)"
echo "Exit code   : $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: processing failed for ${INPUT_FILE}" >&2
    exit $EXIT_CODE
fi

echo "Done: ${BASENAME}"


