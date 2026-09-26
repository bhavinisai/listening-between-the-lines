# Question-type classifier: seed label methodology

`question_type_analysis.py` classifies every extracted question into one of six
types (closed, open, leading, personal, professional, challenge) using a
fine-tuned XLM-R checkpoint. XLM-R needs labeled training examples first; none
existed in this repo, and no suitable public dataset uses this taxonomy, so a
seed training set was built as follows.

## Sampling

`src/sample_question_type_seed.py` draws a stratified sample of 1,280 unique
question texts from the 27,299 questions already extracted by
`question_type_analysis.py --extract_only` (`results/question_classification.csv`),
stratified by (dyad, asker_role) proportional to each stratum's share of the
full set, after deduplicating exact repeated text. Output:
`results/question_type_llm_seed_sample.csv`.

## Labeling

The seed sample was labeled by an LLM (not hand-annotated) applying the
taxonomy definitions and an explicit tie-break priority
(challenge > leading > personal > professional > open/closed by form) to each
question's text alone, with no surrounding transcript context — matching what
the classifier will see at inference time. The sample was split into 4 chunks
of 320 and labeled independently in parallel, then merged into
`results/question_type_seed_labeled.csv` (question_id, text, label).

**This is a methodological limitation to disclose in the writeup**: the
"ground truth" labels the classifier is fine-tuned on are themselves LLM
judgments against the taxonomy, not human-annotated. If inter-annotator
agreement / validity needs to be established for the paper, a
human-labeled subset (e.g. spot-checking 100-150 of these against the LLM
labels, or a fully independent hand-labeled sample) should be added before
treating the classifier's downstream dyad-comparison results as final.

## Fine-tuning

`src/finetune_question_type_xlmr.py` fine-tunes `xlm-roberta-base` on
`results/question_type_seed_labeled.csv` (85/15 stratified train/val split),
saving the checkpoint to `models/xlmr_question_type/` (git-ignored — large
binary, regenerate via the sbatch job rather than committing it).

Run on the cluster via:

```
sbatch sbatch/run_finetune_question_type.sbatch
```

## Full pipeline

```
python src/question_type_analysis.py \
    --episodes results/balanced_200_episodes.csv \
    --transcript_dir data/outputs/whisperx \
    --model_path models/xlmr_question_type \
    --out_dir results
```

This reclassifies all 27,299 previously-extracted questions with the
fine-tuned checkpoint and writes `results/question_classification.csv`
(now with `predicted_type`/`confidence` filled in),
`results/question_type_by_dyad.csv`, and `results/question_type_stats.csv`
(the chi-square + linear-probability dyad tests).
