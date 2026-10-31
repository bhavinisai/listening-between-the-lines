# Sentiment figures — quality audit

Independent recomputation from raw saved files: **42/42 checks passed**.

Title hygiene (no checkpoint names, seed IDs, directory names or methodological text) is enforced inside `make_figures.py` (every title is asserted against a forbidden-pattern list before saving).

- PASS — class order identical to training code (llm_pipeline.CLASSES)
- PASS — valence: validation labels = 256 turns / 18 episodes
- PASS — arousal: validation labels = 256 turns / 18 episodes
- PASS — valence validation labels identical across scaling and ablation experiments
- PASS — validation turn_ids = original validation split (no test / gold rows)
- PASS — valence: retained checkpoint = valence_1783_seed2 and exists
- PASS — valence: recomputed val macro-F1 of retained checkpoint = logged value (0.7945)
- PASS — arousal: retained checkpoint = arousal_5000_seed2 and exists
- PASS — arousal: recomputed val macro-F1 of retained checkpoint = logged value (0.6260)
- PASS — valence_confusion_matrix: plotted counts = recomputed confusion = saved llm_ensemble_check confusion
- PASS — valence_confusion_matrix: row sums = class supports [46, 144, 66]; row percentages sum to 100
- PASS — arousal_confusion_matrix: plotted counts = recomputed confusion = saved llm_ensemble_check confusion
- PASS — arousal_confusion_matrix: row sums = class supports [33, 203, 20]; row percentages sum to 100
- PASS — performance_by_training_size: 18 per-seed scores recomputed; mean / SD (ddof=1) match summary_by_size.csv
- PASS — per_class_performance valence: P/R/F1/support = recomputed = comparison.csv; classes in order ['negative', 'neutral', 'positive']
- PASS — per_class_performance valence: confusion row sums = supports [46, 144, 66]
- PASS — per_class_performance arousal: P/R/F1/support = recomputed = comparison.csv; classes in order ['low', 'medium', 'high']
- PASS — per_class_performance arousal: confusion row sums = supports [33, 203, 20]
- PASS — valence_data_comparison: 15 per-seed scores recomputed; means / SD match ablation summary.csv
- PASS — valence_data_comparison: 5 matched seeds (0-4) per condition
- PASS — valence_ablation_results: plotted bootstrap differences identical to saved bootstrap_differences.csv
- PASS — training sets nested with 1,783 / 3,000 / 5,000 rows
- PASS — training_label_distribution: counts, denominators and rows without majority label recomputed from scaling_train_sets.csv
- PASS — training_label_distribution: denominators match data_verification.json
- PASS — single_model_vs_ensemble: macro-F1 bars = comparison.csv
- PASS — single_model_vs_ensemble: plotted differences / CI = saved bootstrap_{task}.csv
- PASS — single_model_vs_ensemble: neither ensemble selected (decision.json)
- PASS — per_episode_accuracy_change: 18 episodes / 256 turns per task; delta = acc_5000 - acc_1783; improved/worse counts match stability.json
- PASS — performance_by_gender_dyad: n and class counts recomputed from validation labels; n sums to 256 per task
- PASS — performance_by_gender_dyad: plotted values = macro-F1 over ALL three classes (same class definition in every subgroup)
- PASS — performance_by_gender_dyad: subgroups with an absent class are NaN / 'undefined' (no value fabricated); classes with <5 turns flagged unreliable
- PASS — performance_by_gender_dyad: undefined subgroup(s) = arousal MALE->FEMALE
- PASS — earlier_models_test_vs_human_gold: values = saved audit/metrics_full.csv (historical; read only)
- PASS — exactly the 11 expected PNG figures (incl. valence and arousal confusion matrices) (arousal_confusion_matrix, earlier_models_test_vs_human_gold, per_class_performance, per_episode_accuracy_change, performance_by_gender_dyad, performance_by_training_size, single_model_vs_ensemble, training_label_distribution, valence_ablation_results, valence_confusion_matrix, valence_data_comparison)
- PASS — all figure filenames are snake_case
- PASS — no PDF / SVG files in the figure directory
- PASS — all PNGs saved at 300 dpi ([300])
- PASS — all multi-panel figures are 6.5 inches wide; the two single-panel confusion matrices share one size (4.5 in) ([4.5, 6.5])
- PASS — no legacy report-folder naming in the figure scripts or outputs; the legacy source and results folders no longer exist
- PASS — captions: one caption per figure (11)
- PASS — captions contain no 'significant' / p-value claims
- PASS — captions state the evaluation split for every figure
