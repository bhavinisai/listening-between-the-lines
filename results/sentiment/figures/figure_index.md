# Sentiment figures — index

| Figure | File | Title | Suggested placement |
|---|---|---|---|
| 1 | `performance_by_training_size.png` | Performance by Training Size | Results — data scaling |
| 2 | `valence_confusion_matrix.png` | Valence Confusion Matrix | Results — retained model performance |
| 3 | `arousal_confusion_matrix.png` | Arousal Confusion Matrix | Results — retained model performance |
| 4 | `valence_data_comparison.png` | Valence Data Comparison | Results — valence ablation |
| 5 | `valence_ablation_results.png` | Valence Ablation Results | Results — valence ablation |
| 6 | `per_class_performance.png` | Per-Class Performance | Results — retained model performance |
| 7 | `training_label_distribution.png` | Training Label Distribution | Methods — training data |
| 8 | `single_model_vs_ensemble.png` | Single Model vs. Ensemble | Results — ensemble check |
| 9 | `performance_by_gender_dyad.png` | Performance by Gender Dyad | Appendix — subgroup performance |
| 10 | `per_episode_accuracy_change.png` | Per-Episode Accuracy Change | Appendix — robustness across episodes |
| 11 | `earlier_models_test_vs_human_gold.png` | Earlier Models: LLM Test vs. Human Gold | Appendix — label validity (historical) |

All files are in `results/sentiment/figures/` (PNG only). Plotted numbers: `results/sentiment/figures/data/`. Regenerate: `python src/sentiment/figures/make_figures.py && python src/sentiment/figures/write_captions.py && python src/sentiment/figures/verify_figures.py`.
