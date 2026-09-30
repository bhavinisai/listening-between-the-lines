"""
validate_dialogue_act_vs_lexicon.py

Compares the dialogue-act model's 'hold' classification against the same
50-sample directive audit you hand-labeled earlier, to see whether it
agrees better with human judgment than the lexicon-based match did.

Steps:
    1. Load the original directive_audit_sample.csv (with your hand labels)
    2. Relocate each audit row's original turn (+ previous turn) in its
       source transcript JSON, by matching the saved context snippet
    3. Run the dialogue act model on those exact (prev, curr) turn pairs
    4. Compute precision + agreement for the dialogue-act method and
       compare directly against the lexicon's 24% precision

Usage:
    python src/validate_dialogue_act_vs_lexicon.py \
        --audit_csv results/dialogue_acts/directive_audit_sample.csv \
        --out_csv results/dialogue_acts/directive_audit_with_dialogue_act.csv \
        --use_cuda
"""

import argparse
import json

import pandas as pd

LABELS = [
    "acknowledge", "answer", "backchannel", "reply_yes", "exclaim",
    "say", "reply_no", "hold", "ask", "intent", "ask_yes_no",
]

LEXICON_PRECISION = 12 / 50  # from the earlier hand-audit of the lexicon-based 'directive' category


def find_matching_turn(json_path: str, context_snippet: str, anchor_chars: int = 40):
    """
    Relocate the original turn (and its previous turn) in the transcript
    by matching the saved context snippet against segment text.
    Returns None if no match is found (flag for manual check).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    segments = data.get("segments", [])
    snippet_core = context_snippet.strip()[:anchor_chars].lower()
    if not snippet_core:
        return None

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "")
        if snippet_core in text.lower():
            prev_text = segments[i - 1]["text"].strip() if i > 0 else ""
            return {
                "turn_index": i,
                "prev_text": prev_text,
                "curr_text": text.strip(),
            }
    return None


def relocate_audit_rows(audit: pd.DataFrame) -> pd.DataFrame:
    """Relocate every audit row's original turn in its source transcript."""
    prev_texts, curr_texts, found_flags = [], [], []

    for _, row in audit.iterrows():
        match = find_matching_turn(row["episode"], row["context"])
        if match is None:
            prev_texts.append(None)
            curr_texts.append(None)
            found_flags.append(False)
        else:
            prev_texts.append(match["prev_text"])
            curr_texts.append(match["curr_text"])
            found_flags.append(True)

    audit = audit.copy()
    audit["prev_text"] = prev_texts
    audit["curr_text"] = curr_texts
    audit["found"] = found_flags
    return audit


def load_model(use_cuda: bool = False):
    from simpletransformers.classification import ClassificationModel
    # DeBERTa's disentangled attention overflows under fp16 autocast
    # (masked_fill with torch.finfo(float32).min -> at::Half overflow),
    # so force full precision.
    return ClassificationModel(
        "deberta", "diwank/silicone-deberta-pair",
        use_cuda=use_cuda, args={"silent": True, "fp16": False},
    )


def classify_audit_rows(model, audit: pd.DataFrame) -> pd.DataFrame:
    """Run the dialogue act model on the relocated (prev, curr) pairs."""
    found = audit[audit["found"]].copy()
    unfound = audit[~audit["found"]].copy()

    if len(found) == 0:
        print("WARNING: no audit rows could be relocated. Nothing to classify.")
        return audit

    pairs = list(zip(found["prev_text"], found["curr_text"]))
    pairs = [[p, c] for p, c in pairs]

    preds, _ = model.predict(pairs)
    found["dialogue_act_label"] = [LABELS[p] for p in preds]
    found["model_says_directive"] = found["dialogue_act_label"] == "hold"

    unfound["dialogue_act_label"] = None
    unfound["model_says_directive"] = None

    return pd.concat([found, unfound], ignore_index=True)


def main(audit_csv: str, out_csv: str, use_cuda: bool, report_txt: str):
    # Everything appended here is both printed to the console and written,
    # verbatim, to report_txt at the end of the run.
    report_lines: list[str] = []

    def emit(msg: str = ""):
        print(msg)
        report_lines.append(msg)

    audit = pd.read_csv(audit_csv)
    emit(f"Loaded {len(audit)} hand-labeled audit rows from {audit_csv}\n")

    audit = relocate_audit_rows(audit)
    n_found = audit["found"].sum()
    emit(f"Relocated {n_found} / {len(audit)} rows successfully")
    if n_found < len(audit):
        unfound = audit[~audit["found"]]
        emit("Could not relocate these rows — check manually:")
        emit(unfound[["episode", "phrase", "context"]].to_string(index=False))
    emit()

    emit("Loading dialogue act model (diwank/silicone-deberta-pair)...")
    model = load_model(use_cuda=use_cuda)
    emit("Model loaded.\n")

    audit = classify_audit_rows(model, audit)
    audit.to_csv(out_csv, index=False)
    emit(f"Saved per-row results to {out_csv}\n")

    # -------------------------
    # Precision + agreement, on rows that were successfully relocated and classified
    # -------------------------
    scored = audit[audit["found"] & audit["model_says_directive"].notna()].copy()
    scored["is_true_positive"] = scored["is_true_positive"].astype(str).str.strip().str.lower().isin(
        ["yes", "true", "1", "y"]
    )

    tp = ((scored["model_says_directive"] == True) & (scored["is_true_positive"] == True)).sum()
    total_flagged = (scored["model_says_directive"] == True).sum()

    emit("=== Results ===")
    emit(f"Lexicon-based 'directive' precision (earlier audit): {LEXICON_PRECISION:.1%} (12/50)")

    if total_flagged > 0:
        da_precision = tp / total_flagged
        emit(f"Dialogue-act 'hold' precision: {tp}/{total_flagged} = {da_precision:.1%}")
    else:
        emit("Model never predicted 'hold' on this sample — can't compute precision this way.")

    agreement = (scored["model_says_directive"] == scored["is_true_positive"]).mean()
    emit(f"Overall agreement between model's hold/not-hold and human labels: {agreement:.1%}")

    emit()
    emit(scored[["phrase", "context", "is_true_positive", "dialogue_act_label", "model_says_directive"]].to_string(index=False))

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved summary report to {report_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_csv", default="results/dialogue_acts/directive_audit_sample.csv",
                     help="Path to the original hand-labeled directive audit CSV")
    ap.add_argument("--out_csv", default="results/dialogue_acts/directive_audit_with_dialogue_act.csv",
                     help="Where to save the per-row merged results (CSV)")
    ap.add_argument("--report_txt", default="results/dialogue_acts/directive_audit_validation_report.txt",
                     help="Where to save the printed summary report (TXT)")
    ap.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    args = ap.parse_args()
    main(args.audit_csv, args.out_csv, args.use_cuda, args.report_txt)