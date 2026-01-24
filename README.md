# 🎙️ Listening Between The Lines: Analyzing Conversational Flow in Indian Podcasts

### *A Structured Workflow for Extracting, Cleaning, Translating & Analyzing YouTube Podcast Transcripts*

## 📌 **Project Overview**

This project implements a complete pipeline for automatically collecting, cleaning, translating, and performing basic analysis on podcast transcripts from YouTube.
It is designed as a modular, layered architecture that processes raw unstructured content into structured text ready for downstream research.

The system supports:

* Automated transcript extraction from YouTube
* Translation of non-English transcripts
* Cleaning and normalization
* Sentence segmentation
* Basic text analytics
* Batch processing for multiple episodes

This repository contains all scripts, data folders, and sample outputs needed to reproduce the workflow.

## 🗂️ **Repository Structure**

```
podcast_analysis_project/
│
├── data/
│   ├── raw_transcripts/        # Original transcripts (JSON or text)
│   ├── cleaned_transcripts/    # Cleaned/translated text files
│   └── metadata/               # Episode metadata (optional)
│
├── outputs/
│   ├── tables/                  # Word/sentence counts
│   └── samples/                # Extracted question lists
│
├── src/
│   ├── batch_download.py       # Batch transcript downloader
│   ├── convert_json_to_text.py # JSON → TXT converter
│   ├── batch_translate_to_english.py│
│   ├── extract_questions.py
│   ├── text_stats.py        # Word & sentence count generator
│   └── episode_list.txt        # List of YouTube URLs
│
│
└── README.md                   # Project documentation
```

## 🧱 **System Architecture**

The project is built using a 5-layer pipeline:

1. **Input Layer**

   * Stores YouTube URLs for all episodes
   * Provides controlled input for batch processing

2. **Transcript Extraction Layer**

   * Uses `youtube_transcript_api` to download transcripts
   * Saves raw JSON files with time-stamped segments

3. **Translation & Cleaning Layer**

   * Translates Hindi / Hinglish episodes to English
   * Cleans artifacts, spacing, and structural inconsistencies

4. **Preprocessing Layer**

   * Sentence segmentation (NLTK)
   * Tokenization
   * Text normalization
   * Ready for further NLP tasks

5. **Output & Analysis Layer**

   * Extracts user questions
   * Computes word & sentence statistics
   * Produces structured output in `/outputs/`


## ⚙️ **Installation**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Dhanyaravikumarsuchithra/Capstone-Listening-Between-the-Lines.git
cd podcast_analysis_project
```

### 2️⃣ Install dependencies

You should create a virtual environment (recommended):

```bash
pip install youtube-transcript-api deep-translator nltk pandas
```

### 3️⃣ Download NLTK data

```python
import nltk
nltk.download('punkt')
```

---

## 🚀 **How to Run the Pipeline**

### **Step 1 — Add YouTube URLs**

Edit:

```
src/episode_list.txt
```

One URL per line.

---

### **Step 2 — Download Transcripts**

```bash
python src/batch_download.py
```

Raw transcripts will be saved inside:

```
data/raw_transcripts/
```

---

### **Step 3 — Translate (only for Hindi episodes)**

```bash
python src/batch_translate_to_english.py
```

---

### **Step 4 — Clean Transcripts**

```bash
python src/clean_transcript.py
```

---

### **Step 5 — Extract Questions**

```bash
python src/extract_questions.py
```

---

### **Step 6 — Compute Word & Sentence Stats**

```bash
python src/text_stats.py
```

Outputs stored in:

```
outputs/tables/stats
```

---

## 📊 Example Output

### **Word & Sentence Statistics**

```
episode,word_count,sentence_count
ep001_cleaned.txt,6698,312
ep002_cleaned.txt,6624,298
ep003_cleaned.txt,11430,629
...
```

### **Extracted Questions (sample)**

```
What is the most misunderstood thing about you?
Do you think AI poses a global threat?
Why do you believe India is becoming a global talent hub?
...
```

---

## 🧪 **Technologies Used**

| Component             | Library / Tool           |
| --------------------- | ------------------------ |
| Transcript download   | youtube-transcript-api   |
| Translation           | deep-translator          |
| Cleaning              | Python string processing |
| Sentence segmentation | NLTK                     |
| Statistics            | pandas                   |
| Data storage          | Local filesystem         |

---

## 🧾 **Project Goals**

This repository aims to:

* Build a structured and reproducible workflow
* Enable analysis of conversational patterns in podcasts
* Provide early exploratory results for further NLP models
* Form the basis for next-semester research

---

## 🙌 **Team Members**

* Bhavini Sai Mallu
* Sameeksha Rao

