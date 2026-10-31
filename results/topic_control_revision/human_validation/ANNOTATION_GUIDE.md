# Topic initiation and follow-up: annotation guide

**Status: prepared, NOT yet annotated.** No human labels exist yet. Nothing in this folder validates the automated measure until two people have completed the sheets.

## Purpose

The topic-control analysis labels a turn as a **topic initiation** and labels the other speaker's next turn as a **follow-up** using word overlap (TF-IDF similarity). This study checks whether those automatic labels match what a careful human reader judges. Your answers are compared with the automatic labels later. You will **not** see the automatic labels, and you should not try to guess them.

## Who does what

| Step | Person | File |
|---|---|---|
| 1. Read this guide fully, then do the 6 practice examples below | Both | this file |
| 2. Label all 176 items **independently** (no discussion) | Annotator 1 | `annotator_1_sheet.csv` |
| | Annotator 2 | `annotator_2_sheet.csv` |
| 3. Run agreement script | Either | `python src/topic_control_revision/05_validation_agreement.py` |
| 4. Resolve disagreements together (or by a third person) | Both | `adjudication_to_fill.csv` → save as `adjudication_completed.csv` |
| 5. Run accuracy and error scripts | Either | `06_validation_accuracy.py`, `07_validation_error_by_group.py` |

**Do not open** `manifest/validation_manifest.csv` before step 4 is finished: it contains the automatic labels and the speakers' gender and role.

The two sheets contain the same 176 items in a different random order. Open the CSV in Excel, LibreOffice or Google Sheets, fill in the four answer columns and save it as **CSV** with the same file name. Do not edit, delete or reorder the other columns.

## What each row shows

- `context_before`: up to two turns before the item, to show what was being discussed. `[A]` and `[B]` mark the two speakers.
- `A_initiating_turn`: the turn by speaker **A** that you judge for **Q1**.
- `turns_between`: almost always empty. If it is filled, these turns come between A's turn and B's reply.
- `B_next_turn`: speaker **B**'s next turn. You judge it for **Q2**.
- Turns longer than 220 words are shortened, and `[…]` marks the cut. Judge what is shown.
- Speakers are anonymised as A and B. The conversation may still reveal who the host is. Do not let that influence your judgement.
- Some transcripts are machine translations or contain transcription errors. Judge the meaning as best you can, and use `NA` if the text cannot be understood.

## Q1 `Q1_new_topic`: Does A's turn introduce a new topic?

Compare A's turn with `context_before`.

| Code | Use when |
|---|---|
| `NEW` | A moves to a subject that was not being discussed: a new question area, a new story, or a new person or event that is not an elaboration of what came before. |
| `SHIFT` | A stays in the same broad subject but opens a clearly new aspect, sub-question or angle (e.g., from *why you started the company* to *how you funded it*). |
| `SAME` | A continues, answers, elaborates, agrees, disagrees or reacts to what was just being discussed, with no new aspect. **An answer to a question is `SAME`** unless it deliberately moves elsewhere. |
| `UNCLEAR` | You can understand the text but cannot decide between the codes above. |
| `NA` | Unjudgeable: garbled, too short to have content (e.g. "Hmm."), or the context is missing or broken. |

## Q2 `Q2_engagement`: Does B's next turn engage with A's turn?

Judge **regardless of your Q1 answer**: ask whether B takes up what **A said**, even if A did not start a new topic.

| Code | Use when |
|---|---|
| `ENGAGE` | B substantively responds to the content of A's turn: answers it, builds on it, asks about it, disagrees with it, or gives an example of it. Using different words still counts. |
| `MINIMAL` | B only acknowledges or reacts without adding content ("Right, absolutely.", "Wow.", laughter, or repeating A's last words). |
| `NONE` | B ignores A's content and talks about something else or changes the subject. |
| `UNCLEAR` | Understandable, but you cannot decide. |
| `NA` | Unjudgeable (garbled, empty, broken context). |

Shared words are **not** the test. A reply can share many words and still not engage, for example by repeating a stock phrase. It can share no words and engage fully, for example by paraphrasing.

## `confidence_1to3` and `notes`

`3` = confident, `2` = somewhat, `1` = guessing. Use `notes` for anything unusual, such as a diarization error where both turns seem to be the same person, or a translation that is hard to follow.

## Practice examples (constructed for this guide, not from the dataset)

1. Context `[B] …so that's how the first shop opened.` · A: "Let's talk about your childhood. Where did you grow up?" · B: "In Indore, in a joint family of fourteen people." → **Q1 NEW, Q2 ENGAGE**
2. Context `[B] We raised money from friends first.` · A: "And how did you convince the first real investor?" · B: "Honestly, we didn't. He found us after a newspaper article." → **Q1 SHIFT, Q2 ENGAGE** (no shared words, still engagement)
3. Context `[B] What was the hardest year for the business?` · A: "Twenty-nineteen, when the factory flooded." · B: "How long were you shut?" → **Q1 SAME** (A answers the question), **Q2 ENGAGE**
4. A: "I think fitness is really about consistency, not intensity." · B: "Absolutely, absolutely." → **Q1** depends on context, **Q2 MINIMAL**
5. A: "My mother never let us skip breakfast." · B: "Okay. Tell me about the film you just finished." → **Q2 NONE**
6. A: "Uh the the like [inaudible]" → **Q1 NA**; judge Q2 only if B's reply makes sense on its own, otherwise **NA**

Work through them alone first, then compare with the answers above before starting the real sheet.

## Rules

- Work independently. Do not discuss items until both sheets are saved.
- Use only the codes above, in capitals. The scripts reject any other value.
- Budget: roughly 1–1.5 minutes per item, so 3–4.5 hours each. Take breaks; there is no need to finish in one sitting.
- Do not use an LLM or any automated tool to produce or check labels. The point of this study is human judgement.

## Adjudication (step 4)

After `05_validation_agreement.py` runs, `adjudication_to_fill.csv` lists the items where the two annotators disagree on Q1 or Q2, with both answers. For each one, discuss it and enter `final_Q1` and/or `final_Q2` (fill only the question you disagreed on; agreed answers are used automatically), plus `adjudicated_by` and a one-line `rationale`. Save the file as `adjudication_completed.csv` in this folder.

## How the results must be reported

- The sample is **stratified**: 11 items in each of 16 strata (dyad × role × automatic follow-up yes/no × short/long initiating turn), whatever the stratum's size in the corpus. Unweighted precision, recall or accuracy over the 176 items therefore **do not** describe the corpus. `06_validation_accuracy.py` reports design-weighted estimates (weight = stratum size ÷ items drawn) for corpus-level claims, and labels unweighted numbers "sample only".
- Only **automatic initiations** were sampled, so the study estimates how often an automatic initiation is a real topic change (precision). It **cannot** estimate how many real topic changes the method misses (recall).
- Report Cohen's κ for both questions, how many items were `UNCLEAR`/`NA`, and the error comparison by role, gender and dyad (`07_…`). With about 44 items per role × dyad cell, only large differences are detectable.
