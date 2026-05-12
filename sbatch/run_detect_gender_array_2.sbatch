#!/bin/bash
#SBATCH --job-name=gender_detect
#SBATCH --output=logs/gender_detect_%A_%a.out
#SBATCH --error=logs/gender_detect_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --array=36-150  # Adjust based on number of episodes (0 to N-1)

# Change to project root directory
cd /home/sr5868/listening-between-the-lines  # Update with your actual path

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

# Base directories
AUDIO_DIR="data/raw_audio"
JSON_DIR="data/outputs/whisperx"
OUTPUT_DIR="data/outputs/whisperx"

# Get list of all audio files
AUDIO_FILES=(${AUDIO_DIR}/*.wav)

# Get the current file based on array task ID
AUDIO_FILE=${AUDIO_FILES[$SLURM_ARRAY_TASK_ID]}

# Extract base name (e.g., ep_003 from ep_003.wav)
BASENAME=$(basename "${AUDIO_FILE}" .wav)

# Corresponding JSON file (note: no .host_guest suffix)
INPUT_JSON="${JSON_DIR}/${BASENAME}_whisperx_diarized.json"

# Output files
OUTPUT_JSON="${OUTPUT_DIR}/${BASENAME}_whisperx_diarized.gender.json"
OUTPUT_TXT="${OUTPUT_DIR}/${BASENAME}_whisperx_diarized.gender.txt"

echo "========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Processing: ${BASENAME}"
echo "Audio: ${AUDIO_FILE}"
echo "Input JSON: ${INPUT_JSON}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Output TXT: ${OUTPUT_TXT}"
echo "========================================="

# Check if files exist
if [ ! -f "${AUDIO_FILE}" ]; then
    echo "ERROR: Audio file not found: ${AUDIO_FILE}"
    exit 1
fi

if [ ! -f "${INPUT_JSON}" ]; then
    echo "ERROR: JSON file not found: ${INPUT_JSON}"
    exit 1
fi

# Run gender detection (note: using --input-json with hyphen, no --input_txt)
python src/detect_gender_v5.py \
    --audio "${AUDIO_FILE}" \
    --input_json "${INPUT_JSON}" \
    --output_json "${OUTPUT_JSON}" \
    --output_txt "${OUTPUT_TXT}" \
    --speaker_library data/speaker_library.json \
    --match_threshold 0.70

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed: ${BASENAME}"
else
    echo "✗ Failed: ${BASENAME} (exit code: ${EXIT_CODE})"
    exit $EXIT_CODE
fi

echo "========================================="

