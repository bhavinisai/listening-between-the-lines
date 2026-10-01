#!/usr/bin/env python3
"""
compute_bias_features_v2.py

Speaker-level conversational style features per episode, replacing the four
hand-built lexicons in compute_bias_features.py with published resources:

  1. Hyland (2005) metadiscourse lexicon -> hedges, boosters
     (data/lexicons/hyland_2005_hedges_boosters.json; longest-match,
     non-overlapping phrase matching, so "no doubt" is a booster and does
     not also count "doubt" as a hedge). The lexicon's spoken_adjustments
     section excludes spoken fillers ("you know", "I don't know",
     "talk about"), counts "I think" as a hedge, and drops terms that
     audited poorly in speech; disable with --no_spoken_adjustments.
  2. ConvoKit PolitenessStrategies (Danescu-Niculescu-Mizil et al., 2013)
     -> 21 strategies (gratitude, apologizing, deference, please, hedges,
     factuality, direct question, 1st/2nd person, ...), using a spaCy
     dependency parse of each turn. data/lexicons/convokit_adjustments.json
     drops audited-poor marker words (Factuality 'really', Deference 'good')
     and counts Direct_question only in a turn-final sentence ending with
     '?' (a question that hands the floor to the other speaker); disable
     with --no_convokit_adjustments.

Unit of analysis: consecutive WhisperX segments from the same speaker are
merged into one turn before scoring. Segments with no diarized speaker are
dropped. Speech in the first/last --exclude_edges_sec seconds of an episode
(intro/outro monologues, sign-off thank-yous) is excluded.

For every feature the output has:
  <feat>_count      number of occurrences (ConvoKit: number of markers)
  <feat>_per_1k     occurrences per 1,000 words
ConvoKit features also get:
  <feat>_turn_rate  share of the speaker's turns with at least one occurrence

Outputs:
  --out_csv    one row per (episode, speaker), same id/role/gender columns as v1
  --terms_csv  long table of which terms/markers drove each feature, for audits
  --audit_csv  random hits per feature with turn text, for manual precision audit
               (add an is_true_positive column by hand)

Environment: conda activate convokit (convokit 4.x, spaCy 3.x, en_core_web_sm)

Usage:
  python src/conversational_style/compute_bias_features_v2.py \
      --input_glob "data/outputs/whisperx/*_whisperx_diarized.gender.host_guest.json" \
      --out_csv results/features/conversational_style/speaker_features_v2.csv \
      --terms_csv results/features/conversational_style/speaker_feature_terms_v2.csv \
      --audit_csv results/features/conversational_style/audit_sample_v2.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

WORD_RE = re.compile(r"[A-Za-z']+")

# ConvoKit strategy name -> output column prefix
CONVOKIT_FEATURES = {
    "Gratitude": "ck_gratitude",
    "Apologizing": "ck_apologizing",
    "Deference": "ck_deference",
    "Please": "ck_please",
    "Please_start": "ck_please_start",
    "Indirect_(greeting)": "ck_indirect_greeting",
    "Indirect_(btw)": "ck_indirect_btw",
    "HASHEDGE": "ck_hashedge",
    "Hedges": "ck_hedges",
    "Factuality": "ck_factuality",
    "SUBJUNCTIVE": "ck_subjunctive",
    "INDICATIVE": "ck_indicative",
    "Direct_question": "ck_direct_question",
    "Direct_start": "ck_direct_start",
    "1st_person": "ck_1st_person",
    "1st_person_pl.": "ck_1st_person_pl",
    "1st_person_start": "ck_1st_person_start",
    "2nd_person": "ck_2nd_person",
    "2nd_person_start": "ck_2nd_person_start",
    "HASPOSITIVE": "ck_positive",
    "HASNEGATIVE": "ck_negative",
}


# -------------------------
# 1) Hyland lexicon
# -------------------------
def normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


EXCLUDED = "excluded"


def load_hyland(path: str, spoken_adjustments: bool = True) -> Tuple[re.Pattern, Dict[str, str]]:
    """One alternation regex over all terms, longest first, so each span of
    text is matched at most once by the longest applicable term.

    With spoken_adjustments, terms under spoken_adjustments.drop are removed
    from Hyland's lists, phrases under spoken_adjustments.exclude are matched
    but mapped to EXCLUDED (not counted), and hedge_add phrases are counted
    as hedges. Being longer than the words they contain, excluded and added
    phrases take precedence (e.g. "you know" consumes "know")."""
    with open(path, "r", encoding="utf-8") as f:
        lex = json.load(f)

    adj = lex.get("spoken_adjustments", {}) if spoken_adjustments else {}
    drop = adj.get("drop", {})
    sources = [
        (cat, [t for t in lex[cat] if t.lower() not in {d.lower() for d in drop.get(cat, [])}])
        for cat in ("hedge", "booster")
    ]
    if adj:
        sources.append(("hedge", adj.get("hedge_add", {}).get("phrases", [])))
        sources.append((EXCLUDED, adj.get("exclude", {}).get("phrases", [])))

    term_cat: Dict[str, str] = {}
    for cat, terms in sources:
        for term in terms:
            t = term.lower().strip()
            if t in term_cat and term_cat[t] != cat:
                raise ValueError(f"'{t}' is listed as both {term_cat[t]} and {cat}")
            term_cat[t] = cat
    terms = sorted(term_cat, key=len, reverse=True)
    rx = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    return rx, term_cat


def hyland_hits(text: str, rx: re.Pattern, term_cat: Dict[str, str]) -> List[Tuple[str, str]]:
    return [(term_cat[m.group(0).lower()], m.group(0).lower()) for m in rx.finditer(text)]


# -------------------------
# 2) Episode loading
# -------------------------
def load_turns(json_path: str, exclude_edges_sec: float) -> List[Dict[str, Any]]:
    """Merge consecutive same-speaker segments into turns, dropping segments
    without a speaker and those inside the intro/outro window."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = [s for s in data.get("segments", []) if (s.get("text") or "").strip()]
    if not segments:
        return []

    ep_end = max(float(s.get("end") or s.get("start") or 0.0) for s in segments)
    lo, hi = exclude_edges_sec, ep_end - exclude_edges_sec

    turns: List[Dict[str, Any]] = []
    for s in segments:
        speaker = s.get("speaker")
        if not speaker:
            continue
        start = float(s.get("start") or 0.0)
        if exclude_edges_sec > 0 and not (lo <= start <= hi):
            continue
        text = normalize(s["text"].strip())
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
        else:
            turns.append({
                "speaker": speaker,
                "role": (s.get("speaker_role") or "UNKNOWN").upper(),
                "gender": (s.get("gender") or "unknown").lower(),
                "start": start,
                "text": text,
            })
    return turns


# -------------------------
# 3) ConvoKit scoring
# -------------------------
def load_convokit_adjustments(path: Optional[str]) -> Dict[str, Any]:
    """Marker words to drop per strategy, strategies that only count in
    sentences ending with '?', and strategies that only count in the last
    sentence of a turn followed by another speaker's turn. None -> no
    adjustments."""
    if not path:
        return {"drop": {}, "require_q": set(), "require_final": set()}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    drop = {k: {w.lower() for w in v} for k, v in cfg.get("drop_markers", {}).items()
            if not k.startswith("_")}
    require_q = set(cfg.get("require_question_mark", {}).get("strategies", []))
    require_final = set(cfg.get("require_turn_final", {}).get("strategies", []))
    unknown = (set(drop) | require_q | require_final) - set(CONVOKIT_FEATURES)
    if unknown:
        raise ValueError(f"Unknown ConvoKit strategies in {path}: {sorted(unknown)}")
    return {"drop": drop, "require_q": require_q, "require_final": require_final}


def convokit_markers(turns, episode_id, parser, ps, ck_adj) -> List[Dict[str, List[str]]]:
    """Per turn: {strategy: [marker string per occurrence]}, after ck_adj
    (dropped marker words, question-mark requirement) is applied."""
    from convokit import Corpus, Speaker, Utterance

    speakers = {t["speaker"]: Speaker(id=t["speaker"]) for t in turns}
    utts = []
    for i, t in enumerate(turns):
        utts.append(Utterance(
            id=f"{episode_id}__{i}",
            speaker=speakers[t["speaker"]],
            conversation_id=episode_id,
            reply_to=None if i == 0 else f"{episode_id}__{i - 1}",
            text=t["text"],
        ))
    corpus = Corpus(utterances=utts)
    corpus = parser.transform(corpus)
    corpus = ps.transform(corpus, markers=True)

    out = []
    for i in range(len(turns)):
        utt = corpus.get_utterance(f"{episode_id}__{i}")
        meta = utt.meta["politeness_markers"]
        sents = utt.meta.get("parsed") or []
        per_turn = {}
        for strat in CONVOKIT_FEATURES:
            occ = meta.get(f"politeness_markers_=={strat}==", [])
            if strat in ck_adj["require_q"]:
                occ = [m for m in occ
                       if m and m[0][1] < len(sents) and sents[m[0][1]]["toks"]
                       and sents[m[0][1]]["toks"][-1]["tok"] == "?"]
            if strat in ck_adj["require_final"]:
                # Consecutive same-speaker segments are merged, so the next
                # turn (if any) is always the other speaker's.
                occ = [m for m in occ
                       if m and i < len(turns) - 1 and m[0][1] == len(sents) - 1]
            markers = [" ".join(tok for tok, _, _ in m).lower() for m in occ]
            dropped = ck_adj["drop"].get(strat)
            if dropped:
                markers = [m for m in markers if m not in dropped]
            per_turn[strat] = markers
        out.append(per_turn)
    return out


# -------------------------
# 4) Aggregate per speaker
# -------------------------
def process_episode(json_path, args, rx, term_cat, parser, ps, ck_adj, audit):
    episode_id = os.path.splitext(os.path.basename(json_path))[0]
    turns = load_turns(json_path, args.exclude_edges_sec)
    if not turns:
        return [], []

    ck = convokit_markers(turns, episode_id, parser, ps, ck_adj)

    agg = defaultdict(lambda: {
        "word_count": 0, "turn_count": 0,
        "counts": Counter(), "turns_with": Counter(), "terms": Counter(),
    })
    for t, ck_turn in zip(turns, ck):
        key = (t["speaker"], t["role"], t["gender"])
        a = agg[key]
        a["word_count"] += len(WORD_RE.findall(t["text"]))
        a["turn_count"] += 1

        for cat, term in hyland_hits(t["text"], rx, term_cat):
            if cat == EXCLUDED:
                # kept in terms_csv so the exclusions are auditable
                a["terms"][("hy_excluded", term)] += 1
                continue
            feat = f"hy_{cat}"
            a["counts"][feat] += 1
            a["terms"][(feat, term)] += 1
            audit.add(feat, (episode_id, t, term))

        for strat, markers in ck_turn.items():
            feat = CONVOKIT_FEATURES[strat]
            if markers:
                a["counts"][feat] += len(markers)
                a["turns_with"][feat] += 1
                for m in markers:
                    a["terms"][(feat, m)] += 1
                    audit.add(feat, (episode_id, t, m))

    rows, term_rows = [], []
    feats = ["hy_hedge", "hy_booster"] + list(CONVOKIT_FEATURES.values())
    for (speaker, role, gender), a in agg.items():
        wc, tc = a["word_count"], a["turn_count"]
        row = {"episode_id": episode_id, "speaker_id": speaker, "role": role,
               "gender": gender, "word_count": wc, "turn_count": tc}
        for feat in feats:
            c = a["counts"][feat]
            row[f"{feat}_count"] = c
            row[f"{feat}_per_1k"] = c / wc * 1000.0 if wc else 0.0
            if feat.startswith("ck_"):
                row[f"{feat}_turn_rate"] = a["turns_with"][feat] / tc if tc else 0.0
        rows.append(row)
        for (feat, term), c in a["terms"].items():
            term_rows.append({"episode_id": episode_id, "speaker_id": speaker, "role": role,
                              "feature": feat, "term": term, "count": c})
    return rows, term_rows


class AuditReservoir:
    """Uniform random sample of up to n hits per feature (reservoir sampling),
    so memory stays bounded however common a feature is."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.rng = random.Random(seed)
        self.seen: Counter = Counter()
        self.sample: Dict[str, list] = defaultdict(list)

    def add(self, feat: str, hit) -> None:
        self.seen[feat] += 1
        if len(self.sample[feat]) < self.n:
            self.sample[feat].append(hit)
        else:
            j = self.rng.randrange(self.seen[feat])
            if j < self.n:
                self.sample[feat][j] = hit

    def to_frame(self, context_chars: int) -> pd.DataFrame:
        rows = []
        for feat in sorted(self.sample):
            for episode_id, t, term in self.sample[feat]:
                rows.append({"feature": feat, "term": term, "episode_id": episode_id,
                             "speaker_id": t["speaker"], "role": t["role"],
                             "turn_start_sec": round(t["start"], 1),
                             "context": audit_snippet(t["text"], term, context_chars),
                             "is_true_positive": ""})
        return pd.DataFrame(rows)


def locate_hit(text: str, term: str) -> Optional[Tuple[int, int]]:
    """Character span of the first place all words of `term` occur as whole
    words within a short window, in any order (ConvoKit markers can list
    tokens out of order, e.g. 'me forgive' for 'forgive me')."""
    words = [w for w in term.lower().split() if w]
    if not words:
        return None
    word_spans = {w: [m.span() for m in re.finditer(r"\b" + re.escape(w) + r"\b", text, re.I)]
                  for w in set(words)}
    anchor = max(words, key=len)
    for a_start, a_end in word_spans[anchor]:
        lo, hi = a_start, a_end
        ok = True
        for w in words:
            near = [sp for sp in word_spans[w] if abs(sp[0] - a_start) <= 40]
            if not near:
                ok = False
                break
            sp = min(near, key=lambda x: abs(x[0] - a_start))
            lo, hi = min(lo, sp[0]), max(hi, sp[1])
        if ok:
            return lo, hi
    return None


def audit_snippet(text: str, term: str, context_chars: int) -> str:
    """Window of ~context_chars around the hit, with the hit wrapped in [[ ]]."""
    span = locate_hit(text, term)
    if span is None:
        return text[:context_chars] + ("…" if len(text) > context_chars else "")
    lo, hi = span
    marked = text[:lo] + "[[" + text[lo:hi] + "]]" + text[hi:]
    hi += 4
    if len(marked) <= context_chars:
        return marked
    start = max(0, (lo + hi) // 2 - context_chars // 2)
    end = min(len(marked), start + context_chars)
    start = max(0, end - context_chars)
    return ("…" if start > 0 else "") + marked[start:end] + ("…" if end < len(marked) else "")


class _NoAudit:
    def add(self, feat, hit) -> None:
        pass


def main():
    ap = argparse.ArgumentParser(description="Hyland + ConvoKit speaker style features")
    ap.add_argument("--input_glob", required=True, help="Glob for host_guest JSON files")
    ap.add_argument("--out_csv", default="results/features/conversational_style/speaker_features_v2.csv")
    ap.add_argument("--terms_csv", default="results/features/conversational_style/speaker_feature_terms_v2.csv")
    ap.add_argument("--audit_csv", default=None,
                    help="If set, write a random hit sample per feature for manual audit")
    ap.add_argument("--audit_n", type=int, default=50, help="Hits per feature in the audit sample")
    ap.add_argument("--lexicon", default="data/lexicons/hyland_2005_hedges_boosters.json")
    ap.add_argument("--convokit_adjustments", default="data/lexicons/convokit_adjustments.json")
    ap.add_argument("--no_convokit_adjustments", action="store_true",
                    help="Use ConvoKit markers as produced, without convokit_adjustments")
    ap.add_argument("--no_spoken_adjustments", action="store_true",
                    help="Use Hyland's lists as published, without the spoken_adjustments section")
    ap.add_argument("--exclude_edges_sec", type=float, default=60.0,
                    help="Drop speech in the first/last N seconds (intro/outro). 0 keeps everything.")
    ap.add_argument("--min_words", type=int, default=50, help="Drop speakers with fewer words (as v1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import spacy
    from convokit import PolitenessStrategies, TextParser

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        sys.exit(f"No files match {args.input_glob}")

    rx, term_cat = load_hyland(args.lexicon, spoken_adjustments=not args.no_spoken_adjustments)
    ck_adj = load_convokit_adjustments(None if args.no_convokit_adjustments else args.convokit_adjustments)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    parser = TextParser(spacy_nlp=nlp, verbosity=0)
    ps = PolitenessStrategies(verbose=0)

    all_rows, all_terms = [], []
    audit = AuditReservoir(args.audit_n, args.seed) if args.audit_csv else _NoAudit()
    for i, path in enumerate(paths, 1):
        rows, term_rows = process_episode(path, args, rx, term_cat, parser, ps, ck_adj, audit)
        all_rows.extend(rows)
        all_terms.extend(term_rows)
        if i % 25 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {os.path.basename(path)}", flush=True)

    df = pd.DataFrame(all_rows)
    df = df[df["word_count"] >= args.min_words].copy()
    for p in (args.out_csv, args.terms_csv, args.audit_csv):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(df)} speaker-episode rows.")

    terms = pd.DataFrame(all_terms)
    terms.to_csv(args.terms_csv, index=False)
    print(f"[OK] Wrote {args.terms_csv} with {len(terms)} rows.")

    if args.audit_csv:
        audit_df = audit.to_frame(context_chars=240)
        audit_df.to_csv(args.audit_csv, index=False)
        print(f"[OK] Wrote {args.audit_csv} with {len(audit_df)} hits to label.")


if __name__ == "__main__":
    main()
