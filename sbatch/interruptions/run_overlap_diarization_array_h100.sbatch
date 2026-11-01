#!/bin/bash
#SBATCH -p sporc-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=overlap_diar
#SBATCH --output=logs/overlap_diar_%A_%a.out
#SBATCH --error=logs/overlap_diar_%A_%a.err
#SBATCH --array=51-330

# Overlap-preserving diarization for episode N = line N of episode_list.txt (ep_NNN).
# If data/raw_audio/ep_NNN.wav is missing it is downloaded first.
# With DELETE_AUDIO=1 (sbatch --export=ALL,DELETE_AUDIO=1 ...), audio this job
# downloaded is deleted after diarization, to save disk space.
# Pilot first:  sbatch --array=331-335 sbatch/interruptions/run_overlap_diarization_array.sbatch

set -euo pipefail

cd /home/sr5868/listening-between-the-lines
mkdir -p logs data/outputs/overlap

source /home/sr5868/miniconda3/etc/profile.d/conda.sh
conda activate whisperx_cudnn8

EP=$(printf "ep_%03d" "${SLURM_ARRAY_TASK_ID}")
AUDIO_FILE="data/raw_audio/${EP}.wav"
OUT_FILE="data/outputs/overlap/${EP}_overlap_diarization.json"
DOWNLOADED=0

if [ -f "${OUT_FILE}" ]; then
  echo "Already done: ${OUT_FILE}"
  exit 0
fi

if [ ! -f "${AUDIO_FILE}" ]; then
  URL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" episode_list.txt)
  echo "Downloading ${EP} from ${URL}"
  python src/transcript_creation/youtube_audio_scraper.py --url "${URL}" --output data/raw_audio --filename "${EP}"
  DOWNLOADED=1
fi

echo "Processing: ${AUDIO_FILE} on $(hostname)"
python src/interruptions/overlap_diarization.py "${AUDIO_FILE}" \
  --output_dir data/outputs/overlap \
  --hf_token "$HF_TOKEN"

if [ "${DOWNLOADED}" = "1" ] && [ "${DELETE_AUDIO:-0}" = "1" ]; then
  rm -f "${AUDIO_FILE}"
  echo "Deleted downloaded audio ${AUDIO_FILE}"
fi

echo "Done: ${EP}"
