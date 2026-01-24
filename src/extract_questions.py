import os
import re
from nltk.tokenize import sent_tokenize

INPUT_DIR = "data/cleaned_transcripts"
OUTPUT_DIR = "outputs/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUESTION_PATTERNS = [
    r"\bwhat\b",
    r"\bwhy\b",
    r"\bhow\b",
    r"\bwhen\b",
    r"\bwhere\b",
    r"\bwho\b",
    r"\bdo you\b",
    r"\bcan you\b",
    r"\bis it\b",
    r"\bwould\b",
    r"\bcould\b",
    r"\bshould\b",
]
pattern = re.compile("|".join(QUESTION_PATTERNS), re.IGNORECASE)

def clean_sentence(s):
    return " ".join(s.split()).strip()

def is_question(sentence):
    s = sentence.lower().strip()

    
    if s.endswith("?"):
        return True

   
    if pattern.search(s.split()[0]):
        return True

    
    if pattern.search(s) and len(s) < 140:
        return True

    return False

def extract_questions_from_text(text):
    
    text = " ".join(text.splitlines())

    
    sentences = sent_tokenize(text)

    questions = []
    for s in sentences:
        cs = clean_sentence(s)
        if is_question(cs):
            questions.append(cs)

    return questions


for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".txt"):
        continue

    in_path = os.path.join(INPUT_DIR, file)

    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()

    questions = extract_questions_from_text(text)

    out_path = os.path.join(OUTPUT_DIR, file.replace(".txt", "_questions_v2.txt"))
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q + "\n")

    print(f"{file}: extracted {len(questions)} questions → {out_path}")