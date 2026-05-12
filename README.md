# 🎙️ Listening Between The Lines: Analyzing Gendered Conversational Dynamics in Indian Podcasts

### *An End-to-End Pipeline for Audio Collection, Transcription, Speaker Diarization, Gender Detection, and Conversational Analysis*

---

## 📌 Project Overview

This project implements a complete pipeline for automatically collecting, processing, and analyzing podcast audio from YouTube. It focuses on studying gendered conversational dynamics in English-language Indian podcasts by extracting structured transcripts with speaker roles, gender labels, and conversational features.

The system supports:

- Automated audio collection from YouTube
- Speech transcription with word-level alignment (WhisperX)
- Speaker diarization (identifying who speaks when)
- Hybrid gender detection via voice embedding matching and acoustic classification
- Host/Guest role labeling using speaker identity and LLM-based inference
- Dataset-level statistics and conversational feature extraction
- Batch processing across hundreds of episodes via SLURM

---

## 🗂️ Repository Structure

```
listening-between-the-lines/
│
├── data/
│   ├── speaker_library.json            # Enrolled host voice embeddings
│   ├── raw_audio/                  # Episode .wav files (not tracked by git)
│   │   └── host/                   # Host enrollment audio clips
│   └── outputs/
│       └── whisperx/               # All transcript JSON and TXT outputs
│
├── src/
│   ├── youtube_audio_scraper.py    # Download audio from YouTube
│   ├── diarizze_whisperx_gpu.py    # WhisperX transcription + diarization
│   ├── build_speaker_library.py    # Enroll known hosts into voice library
│   ├── detect_gender_v4.py         # Hybrid gender detection
│   ├── labeled_transcript_v12.py   # Host/Guest role labeling
│   ├── dataset_stats.py            # Dataset-level statistics
│   └── compare_audio.py            # Audio property comparison utility
│
├── sbatch/                         # SLURM batch job scripts
│   ├── run_whisperx_array.sbatch
│   ├── run_detect_gender_array.sbatch
│   └── run_host_guest_labeling.sbatch
│
├── logs/                           # SLURM job logs
├── episode_list.txt            # YouTube URLs for batch download
├── requirements.txt                # Python dependencies
├── audio_files.txt 
└── README.md
```

---

## 🧱 Pipeline Overview

The project is built as a 5-stage pipeline:

```
YouTube Audio → WhisperX Transcription → Gender Detection → Host/Guest Labeling → Analysis
```

![Methodology Diagram](methodology_diagram.png)

### Stage 1 — Audio Collection
Downloads podcast episodes as `.wav` files from YouTube using `yt-dlp`.

### Stage 2 — Transcription & Speaker Diarization
Uses WhisperX (`large-v2`) for speech-to-text with word-level alignment, and pyannote for speaker diarization — producing timestamped transcripts with SPEAKER_00/SPEAKER_01 labels.

### Stage 3 — Gender Detection
Uses a hybrid approach:
1. Each speaker's audio is extracted and converted to a voice embedding using pyannote's speaker embedding model
2. The embedding is compared against a pre-built library of known host voice embeddings using cosine similarity
3. If a confident match is found → gender assigned from library
4. If no match (guest) → falls back to inaSpeechSegmenter, a CNN-based acoustic classifier

### Stage 4 — Host/Guest Labeling
Identifies which speaker is the HOST and which is the GUEST using:
1. Speaker library match (known host → HOST)
2. Groq LLM inference (fallback)
3. Heuristic detection (final fallback)

### Stage 5 — Analysis
Extracts speaker-level features (speaking time, turn count, dominance ratio) and computes dataset statistics across all episodes.

---

## 📊 Dataset

- **Source:** English-language Indian podcasts from YouTube
- **Size:** ~300 episodes
- **Format:** Two-speaker conversations (host + guest)
- **Known Hosts:** 5 recurring hosts enrolled in the speaker library
- **Language:** English

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dhanyaravikumarsuchithra/Capstone-Listening-Between-the-Lines.git
cd listening-between-the-lines
```

### 2. Create and activate conda environment

```bash
conda create -n whisperx_cudnn8 python=3.10
conda activate whisperx_cudnn8
```

### 3. Install PyTorch with CUDA

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Set environment variables

```bash
export HF_TOKEN="your_huggingface_token"        # pyannote models
export YOUTUBE_API_KEY="your_youtube_api_key"   # YouTube Data API v3
export GROQ_API_KEY="your_groq_api_key"         # Host/Guest labeling fallback
```

---

## 🚀 How to Run the Pipeline

### Step 1 — Download Audio

**Single episode:**
```bash
python src/youtube_audio_scraper.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output data/raw_audio \
    --filename ep_001
```

**Batch from URL list:**
```bash
python src/youtube_audio_scraper.py \
    --id-file podcast_episodes.txt \
    --output data/raw_audio
```

> If YouTube bot detection blocks downloads on the server, download locally and transfer:
> ```bash
> scp ep_001.wav user@server:/path/to/data/raw_audio/
> ```

---

### Step 2 — Transcription & Diarization

```bash
python src/diarizze_whisperx_gpu.py data/raw_audio/ep_001.wav \
    --output_dir data/outputs/whisperx \
    --model large-v2 \
    --batch_size 16 \
    --hf_token "$HF_TOKEN" \
    --language en
```

**SLURM batch:**
```bash
sbatch sbatch/run_whisperx_array.sbatch
```

---

### Step 3 — Build Speaker Library (One-Time)

Extract a 30–60 second clip of each host:
```bash
ffmpeg -i data/raw_audio/ep_001.wav -ss 00:02:30 -to 00:03:30 data/raw_audio/host/host_name.wav
```

Enroll hosts into the library:
```bash
python src/build_speaker_library.py \
    --hosts_dir data/raw_audio/host/ \
    --output_library data/speaker_library.json \
    --metadata data/raw_audio/host/metadata.csv
```

`metadata.csv` format:
```
filename,name,gender
host_name.wav,Host Name,male
```

---

### Step 4 — Gender Detection

```bash
python src/detect_gender_v4.py \
    --audio data/raw_audio/ep_001.wav \
    --input_json data/outputs/whisperx/ep_001_whisperx_diarized.json \
    --output_json data/outputs/whisperx/ep_001_whisperx_diarized.gender.json \
    --output_txt data/outputs/whisperx/ep_001_whisperx_diarized.gender.txt \
    --speaker_library data/speaker_library.json \
    --match_threshold 0.70
```

**SLURM batch:**
```bash
sbatch sbatch/run_detect_gender_array.sbatch
```

---

### Step 5 — Host/Guest Labeling

```bash
python src/labeled_transcript_v12.py \
    --input data/outputs/whisperx/ep_001_whisperx_diarized.gender.json \
    --out_dir data/outputs/whisperx/ \
    --speaker_key speaker
```

**SLURM batch with dependency on gender detection:**
```bash
GENDER_JOB=$(sbatch sbatch/run_detect_gender_array.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$GENDER_JOB sbatch/run_host_guest_labeling.sbatch
```

---

### Step 6 — Dataset Statistics

```bash
python src/dataset_stats.py \
    --json_dir data/outputs/whisperx/ \
    --output_csv data/outputs/dataset_stats.csv
```

**Output includes:**
- Total episodes processed
- Episodes by host gender (male / female / unknown)
- Guest gender breakdown per host gender
- Episode count per known host

---

## 🧪 Technologies Used

| Component | Library / Tool |
|---|---|
| Audio download | yt-dlp, google-api-python-client |
| Transcription | WhisperX 3.8.2, faster-whisper 1.2.1 |
| Speaker diarization | pyannote-audio 4.0.4 |
| Voice embeddings | pyannote/embedding |
| Acoustic gender detection | inaSpeechSegmenter 0.8.0 |
| Host/Guest labeling | Groq API (llama-3.3-70b-versatile) |
| Deep learning | PyTorch 2.8.0+cu128 |
| NLP | transformers 4.57.6 |
| Numerical | numpy 2.2.6, scipy 1.15.3 |
| Batch processing | SLURM |

---

## 🖥️ HPC / SLURM Notes

- **Partition:** `tier3`
- **GPU:** NVIDIA A100
- **Environment:** `whisperx_cudnn8` (Python 3.10, CUDA 12.8)
- Array jobs are used to process episodes in parallel
- Jobs can be chained using `--dependency=afterok:JOB_ID`

---

## ⚠️ Known Limitations

- YouTube bot detection may block server-side downloads — use local download + `scp` as a workaround
- Speakers with androgynous or transitional-range voices (~150–165 Hz) may be classified as `unknown` by inaSpeechSegmenter
- Speaker library matching requires a cosine similarity threshold of 0.70 — voices with different recording conditions may score lower
- pyannote diarization occasionally merges two speakers into one label, especially during crosstalk

---

## 🙌 Team Members

- Bhavini Sai Mallu
- Sameeksha Rao

---

## 🤝 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) — Bain et al.
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — Bredin et al.
- [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) — INA
- [Groq](https://console.groq.com) — LLM inference API
