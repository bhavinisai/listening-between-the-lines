#!/bin/bash
#SBATCH --job-name=detect_gender
#SBATCH --output=logs/detect_gender_%j.out
#SBATCH --error=logs/detect_gender_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=sporc
#SBATCH --array=1-24%5

# Load modules (comment out if not needed)
# module load python/3.11
# module load cuda/12.1

# Activate conda environment
conda activate lbtll-py311

# Set directories
AUDIO_DIR="data/raw_audio"
JSON_DIR="data/outputs"
OUTPUT_DIR="data/outputs/gender"
LOG_DIR="logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Get list of audio files
mapfile -t audio_files < <(find "$AUDIO_DIR" -name "*.wav" | sort)

# Calculate array index
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Check if index is valid
if [ $INDEX -ge ${#audio_files[@]} ]; then
    echo "No file for index $INDEX (total files: ${#audio_files[@]})"
    exit 0
fi

# Get current file
AUDIO_FILE="${audio_files[$INDEX]}"
BASENAME=$(basename "$AUDIO_FILE" .wav)
JSON_FILE="$JSON_DIR/${BASENAME}.diarized.json"

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "JSON file not found: $JSON_FILE"
    exit 1
fi

# Set output files
OUTPUT_JSON="$OUTPUT_DIR/${BASENAME}.diarized.gender.json"
OUTPUT_TXT="$OUTPUT_DIR/${BASENAME}.diarized.gender.txt"

# Check if already processed
if [ -f "$OUTPUT_JSON" ] && [ -f "$OUTPUT_TXT" ]; then
    echo "Already processed: $BASENAME"
    exit 0
fi

echo "Processing: $BASENAME"
echo "Audio: $AUDIO_FILE"
echo "JSON: $JSON_FILE"
echo "Output JSON: $OUTPUT_JSON"
echo "Output TXT: $OUTPUT_TXT"

# Run gender detection
python src/detect_gender_simple.py \
    --audio "$AUDIO_FILE" \
    --input-json "$JSON_FILE" \
    --output-json "$OUTPUT_JSON" \
    --output-txt "$OUTPUT_TXT"

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ Success: $BASENAME"
else
    echo "❌ Failed: $BASENAME"
    exit 1
fi

