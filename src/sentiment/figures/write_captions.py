#!/usr/bin/env python3
"""
Writes results/sentiment/figures/captions.md and figure_index.md. Numbers quoted in captions are read from the plotted-data
tables in results/sentiment/figures/data/ and the saved result files, not typed by hand.

    python src/sentiment/figures/write_captions.py
"""
import json
import re

import pandas as pd

OUT, D = "results/sentiment/figures", "results/sentiment/figures/data"
SC, ST, EN = "results/sentiment/llm_scaling_exp", "results/sentiment/llm_scaling_stratum", "results/sentiment/llm_ensemble_check"


def s(v, d=3):
    t = f"{v:+.{d}f}" if abs(v) >= 0.0005 or v == 0 else f"{v:+.4f}"
    return t.replace("-", "−")


f1 = pd.read_csv(f"{D}/performance_by_training_size.csv")
mean = lambda t, n: f"{f1[(f1.task == t) & (f1['size'] == n)].val_macro_f1.mean():.3f}"
abl = pd.read_csv(f"{D}/valence_data_comparison.csv")
am = {c: abl[abl.condition == c] for c in ("base", "A_targeted", "B_random")}
B = pd.read_csv(f"{D}/valence_ablation_results.csv")
b = lambda comp, met: B[(B.comparison == comp) & (B.metric == met)].iloc[0]
ci = lambda comp, met: f"{s(b(comp, met)['diff'])} [{s(b(comp, met).ci_lo)}, {s(b(comp, met).ci_hi)}]"
pc = pd.read_csv(f"{D}/per_class_performance.csv")
sup = lambda t: ", ".join(f"{r.cls.lower()} {r.support}" for r in pc[pc.task == t].itertuples())
comp = pd.read_csv(f"{EN}/comparison.csv")
cm = lambda t, k: comp[(comp.task == t) & (comp.model == k)].macro_f1.iloc[0]
b7 = pd.read_csv(f"{D}/single_model_vs_ensemble_differences.csv")
e7 = lambda t, ref: b7[(b7.task == t) & (b7.ensemble_minus == ref)].iloc[0]
e7s = lambda t, ref: f"{s(e7(t, ref)['diff'])} [{s(e7(t, ref).ci_lo)}, {s(e7(t, ref).ci_hi)}]"
st = {r["task"]: r for r in json.load(open(f"{SC}/stability.json"))}
dy = pd.read_csv(f"{D}/performance_by_gender_dyad.csv")
hist = pd.read_csv(f"{D}/earlier_models_test_vs_human_gold.csv")
h = lambda t, e: hist[(hist.task == t) & (hist.eval_set == e)].iloc[0]
tr = pd.read_csv(f"{D}/training_label_distribution.csv")
trn = lambda t, n: tr[(tr.task == t) & (tr["size"] == n)].labeled_rows.iloc[0]


ORDER = ["performance_by_training_size", "valence_confusion_matrix", "arousal_confusion_matrix", "valence_data_comparison", "valence_ablation_results",
         "per_class_performance", "training_label_distribution", "single_model_vs_ensemble", "performance_by_gender_dyad", "per_episode_accuracy_change",
         "earlier_models_test_vs_human_gold"]
N = {stem: i + 1 for i, stem in enumerate(ORDER)}

SPLIT = "the fixed validation set (256 turns from 18 validation episodes, disjoint from the training episodes; labels are the three-pass LLM majority)"
undef = dy[dy.status.str.startswith("undefined")]
unrel = dy[dy.status.str.startswith("unreliable")]

NAMES = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}


def _classes(r, strict):
    cnt = json.loads(r.class_counts)
    return [(NAMES[r.task][i], c) for i, c in enumerate(cnt) if (c == 0 if strict else 0 < c < 5)]


_lab = lambda r: r.dyad.replace("->", " → ").title()
undef_txt = "; ".join(f"{r.task}, {_lab(r)}: no {n}-{r.task} turns" for r in undef.itertuples() for n, _ in _classes(r, True)) or "none"
unrel_txt = "; ".join(f"{r.task}, {_lab(r)}: only {c} {n}-{r.task} turn{'s' if c != 1 else ''}" for r in unrel.itertuples() for n, c in _classes(r, False)) or "none"
dy_n = {r.dyad: r.n for r in dy[dy.task == "valence"].itertuples()}
dyad_txt = ", ".join(f"{k.replace('->', ' → ').title()} n = {v}" for k, v in dy_n.items())

C = [
    ("performance_by_training_size", "Performance by Training Size", "Results — data scaling",
     (f"**Figure 1. Performance by training size.** Validation macro-F1 of valence (left) and arousal (right) models trained on nested sets of 1,783, 3,000 and 5,000 "
      f"LLM-labeled turns, evaluated on {SPLIT}. Open circles show the three individual training seeds; filled circles with bars show the mean ± one standard deviation across "
      f"seeds. Mean macro-F1 was {mean('valence', 1783)}, {mean('valence', 3000)} and {mean('valence', 5000)} for valence and {mean('arousal', 1783)}, {mean('arousal', 3000)} and "
      f"{mean('arousal', 5000)} for arousal. With only three seeds the standard deviation is itself imprecise; the vertical axes are truncated to make seed-to-seed spread visible. "
      f"Each model is the best-validation epoch of its run, so the scores are optimistic development estimates, not held-out test performance."),
     "performance_by_training_size.csv"),
    ("valence_confusion_matrix", "Valence Confusion Matrix", "Results — retained model performance",
     (f"**Figure 2. Valence confusion matrix.** Confusion matrix of the retained valence model on {SPLIT}. Rows are true (LLM-majority) labels and columns are predicted labels; each cell "
      f"shows the number of validation turns and, in parentheses, the percentage of that true class (row); shading shows the same percentage. Class supports: {sup('valence')}. The matrix describes a "
      f"single model, so no error bars are drawn; rare-class rows rest on few turns, and the model was selected using these same validation data."),
     "valence_confusion_matrix.csv"),
    ("arousal_confusion_matrix", "Arousal Confusion Matrix", "Results — retained model performance",
     (f"**Figure 3. Arousal confusion matrix.** Confusion matrix of the retained arousal model on {SPLIT}. Rows are true (LLM-majority) labels and columns are predicted labels; each cell "
      f"shows the number of validation turns and, in parentheses, the percentage of that true class (row); shading shows the same percentage. Class supports: {sup('arousal')}. The matrix describes a "
      f"single model, so no error bars are drawn; the low- and high-arousal rows rest on 33 and 20 turns, and the model was selected using these same validation data."),
     "arousal_confusion_matrix.csv"),
    ("valence_data_comparison", "Valence Data Comparison", "Results — valence ablation",
     (f"**Figure 2. Valence data comparison.** Validation macro-F1 (left) and negative-class recall (right) for the baseline valence model (1,782 labeled turns) and for models trained "
      f"with 1,106 additional LLM-labeled turns drawn either from targeted rare-class strata (+ Targeted) or from the random stratum (+ Random). The two additions are matched in size and "
      f"dyad counts. Bars show the mean over five matched seeds (macro-F1 {am['base'].macro_f1.mean():.3f}, {am['A_targeted'].macro_f1.mean():.3f}, {am['B_random'].macro_f1.mean():.3f}; "
      f"negative recall {am['base'].neg_recall.mean():.3f}, {am['A_targeted'].neg_recall.mean():.3f}, {am['B_random'].neg_recall.mean():.3f}); error bars show ± one standard deviation across seeds "
      f"and open circles show individual seeds. Evaluation: {SPLIT}. The added subsets differ by design in class mix (targeted ≈ 27% negative, random ≈ 10.5%) and only partly in episode "
      f"composition; bars start at zero, so small differences appear small."),
     "valence_data_comparison.csv"),
    ("valence_ablation_results", "Valence Ablation Results", "Results — valence ablation",
     (f"**Figure 3. Valence ablation results.** Differences in validation macro-F1, negative-class recall and negative-class precision between the three conditions of Figure {N['valence_data_comparison']}. Points are mean "
      f"bootstrap differences and bars are 95% percentile confidence intervals; the dashed line marks no difference. The resampling unit is the validation episode (18 episodes, 2,000 resamples), "
      f"and within each resample every condition's metric is averaged over its five seeds, so the intervals reflect uncertainty over episodes and not seed variability. Macro-F1: targeted − baseline "
      f"{ci('A_targeted - base', 'macro_f1')}; random − baseline {ci('B_random - base', 'macro_f1')}; targeted − random {ci('A_targeted - B_random', 'macro_f1')}. Negative recall: targeted − baseline "
      f"{ci('A_targeted - base', 'neg_r')}; targeted − random {ci('A_targeted - B_random', 'neg_r')}. Evaluation: {SPLIT}. Only 18 episodes and 46 negative turns underlie the intervals, "
      f"several of which lie close to zero; no multiplicity correction was applied and the intervals are descriptive."),
     "valence_ablation_results.csv"),
    ("per_class_performance", "Per-Class Performance", "Results — retained model performance",
     (f"**Figure 4. Per-class performance.** Precision, recall and F1 for each class of the retained valence model (left) and retained arousal model (right), evaluated on {SPLIT}. "
      f"Class supports are: valence {sup('valence')}; arousal {sup('arousal')}. Each panel shows a single model, so no error bars are drawn. Scores for the rare classes rest on 20–46 validation turns "
      f"and carry substantial uncertainty; the models were selected using these same validation data."),
     "per_class_performance.csv"),
    ("training_label_distribution", "Training Label Distribution", "Methods — training data",
     (f"**Figure 5. Training label distribution.** Number of training examples per class in the nested training sets of 1,783, 3,000 and 5,000 turns, for valence (left) and arousal (right); "
      f"labels are the three-pass LLM majority. Only rows with a two-of-three majority label for the task are counted: for valence N = {trn('valence', 1783):,}, {trn('valence', 3000):,} and "
      f"{trn('valence', 5000):,}; for arousal N = {trn('arousal', 1783):,}, {trn('arousal', 3000):,} and {trn('arousal', 5000):,}; the remaining rows have no majority label and were excluded for "
      f"that task. All rows come from the 121 training episodes. The additional turns were selected by a stratified design (targeted rare-class candidates plus a random sample), so the "
      f"class proportions reflect that sampling and not corpus prevalence. This figure describes training data and involves no evaluation."),
     "training_label_distribution.csv"),
    ("single_model_vs_ensemble", "Single Model vs. Ensemble", "Results — ensemble check",
     (f"**Figure 6. Single model versus ensemble.** Top: validation macro-F1 of the retained single model, the average over the individual seed models, and the equal-weight probability ensemble of "
      f"those seed models (valence: five seeds of the 1,783-example set, {cm('valence', 'selected single'):.3f}, {cm('valence', 'mean single (avg of seeds)'):.3f}, {cm('valence', 'ensemble'):.3f}; "
      f"arousal: three seeds of the 5,000-example set, {cm('arousal', 'selected single'):.3f}, {cm('arousal', 'mean single (avg of seeds)'):.3f}, {cm('arousal', 'ensemble'):.3f}). Bottom: ensemble minus "
      f"comparator macro-F1 with 95% percentile bootstrap confidence intervals (resampling unit: the 18 validation episodes, 2,000 resamples); the dashed line marks no difference. Valence: "
      f"ensemble − single model {e7s('valence', 'selected single')}, ensemble − seed average {e7s('valence', 'mean single')}; arousal: {e7s('arousal', 'selected single')} and "
      f"{e7s('arousal', 'mean single')}. Evaluation: {SPLIT}. The ensemble was not selected; the retained models remain the single models. The single model was the best seed on these validation data, "
      f"so it is a favourable comparator, and the ensemble members share training data, so their errors are correlated."),
     "single_model_vs_ensemble_macro_f1.csv; single_model_vs_ensemble_differences.csv"),
    ("performance_by_gender_dyad", "Performance by Gender Dyad", "Appendix — subgroup performance",
     (f"**Figure 7. Performance by gender dyad.** Validation macro-F1 of the retained valence (left) and arousal (right) models for each gender dyad, as coded in the dataset (e.g., Male → Female); "
      f"subgroup sizes are shown as n. Evaluation: {SPLIT}; valence subgroup sizes: {dyad_txt}. Macro-F1 is computed over all three classes in every subgroup, so values are comparable in definition "
      f"across subgroups. It is undefined when a class has no true examples in the subgroup and is then not plotted ({undef_txt}"
      f"). Open markers flag subgroups in which at least one class has fewer than five true turns, where the score is unreliable ({unrel_txt}). Per-subgroup class counts, flags, and the earlier "
      f"present-class-only values are in the companion table. No confidence intervals were estimated; differences between dyads are descriptive and cannot be read as evidence of bias or its absence."),
     "performance_by_gender_dyad.csv"),
    ("per_episode_accuracy_change", "Per-Episode Accuracy Change", "Appendix — robustness across episodes",
     (f"**Figure 8. Per-episode accuracy change.** Change in accuracy for each of the 18 validation episodes when the model is trained on 5,000 instead of 1,783 examples, for valence (left) and "
      f"arousal (right); episodes are ranked by change and accuracy is averaged over three seeds per training size. Valence: {st['valence']['episodes_improved']} episodes improved, "
      f"{st['valence']['episodes_worse']} worsened, {st['valence']['episodes_same']} unchanged; arousal: {st['arousal']['episodes_improved']}, {st['arousal']['episodes_worse']} and "
      f"{st['arousal']['episodes_same']}. Evaluation: {SPLIT}. Individual episodes contain few turns, so per-episode values are noisy; the figure shows the consistency of direction, not precise "
      f"per-episode effects."),
     "per_episode_accuracy_change.csv"),
    ("earlier_models_test_vs_human_gold", "Earlier Models: LLM Test vs. Human Gold", "Appendix — label validity (historical)",
     (f"**Figure 9. Earlier models on the LLM-majority test set and the human-gold sheet (historical).** Macro-F1 of earlier models (single seed) for valence (left) and arousal (right) on the held-out "
      f"LLM-majority test set ({int(h('valence', 'test').n_turns)} turns for valence and {int(h('arousal', 'test').n_turns)} for arousal, {int(h('valence', 'test').n_episodes)} episodes) and on the human-gold sheet ({int(h('valence', 'gold').n_turns)} turns, "
      f"{int(h('valence', 'gold').n_episodes)} episodes). Points show macro-F1 (valence {h('valence', 'test').macro_f1:.3f} and {h('valence', 'gold').macro_f1:.3f}; arousal "
      f"{h('arousal', 'test').macro_f1:.3f} and {h('arousal', 'gold').macro_f1:.3f}); bars are 95% bootstrap confidence intervals resampling episodes of the respective set; dashed lines show the macro-F1 "
      f"of always predicting the majority class. These values come from previously saved results of earlier model configurations and training data; they are not the retained models and are not "
      f"comparable with the validation results in Figures 1–{N['per_episode_accuracy_change']}. The gap between the two sets reflects weak agreement between the LLM-majority labels and the single-annotator human labels."),
     "earlier_models_test_vs_human_gold.csv"),
]

md = ["# Sentiment figures — captions", "",
      "All figures are generated by `src/sentiment/figures/make_figures.py` from saved results only (no training, no new experiments, no tuning). PNG, 300 dpi, 6.5 in wide. "
      "Plotted numbers are in `data/`. Retained models: valence trained on 1,783 examples and arousal trained on 5,000 examples (development-selected on the validation set).", ""]
assert [c[0] for c in C] == ORDER
for i, (stem, title, section, cap, src) in enumerate(C, 1):
    cap = re.sub(r"^\*\*Figure \d+\.", f"**Figure {i}.", cap)
    md += [f"### {title}", "", cap, "", f"*File:* `{stem}.png` · *Suggested placement:* {section} · *Source data:* `data/{src}`", ""]
md += ["### Global limitations", "",
       "1. Unless a caption states otherwise, performance is development-validation performance on 256 turns from 18 episodes; each run's checkpoint is its best-validation epoch, so scores are optimistic and are not held-out test estimates.",
       "2. Labels are LLM-annotator majorities (three passes of one model family); agreement with the single-annotator human-gold labels was weak in earlier analyses.",
       "3. Few seeds (3–5) were used; bootstrap intervals resample only the 18 validation episodes. No hypothesis tests were run and no multiplicity correction was applied; intervals are descriptive."]
open(f"{OUT}/captions.md", "w").write("\n".join(md) + "\n")

idx = ["# Sentiment figures — index", "", "| Figure | File | Title | Suggested placement |", "|---|---|---|---|"]
for i, (stem, title, section, cap, src) in enumerate(C, 1):
    idx.append(f"| {i} | `{stem}.png` | {title} | {section} |")
idx += ["", "All files are in `results/sentiment/figures/` (PNG only). Plotted numbers: `results/sentiment/figures/data/`. Regenerate: "
        "`python src/sentiment/figures/make_figures.py && python src/sentiment/figures/write_captions.py && python src/sentiment/figures/verify_figures.py`."]
open(f"{OUT}/figure_index.md", "w").write("\n".join(idx) + "\n")
print("captions and index written")
