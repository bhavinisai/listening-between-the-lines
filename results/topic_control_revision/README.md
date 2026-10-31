# Topic-control revision (2026-10-29)

This revises the Report 1 topic-control analysis without changing it. The originals in `results/topic_control/`, `results/figures/topic_control/` and `src/topic_control/` are untouched. Start with **`report_revision_guide.md`**, which has the exact report changes, captions and report-ready text.

## What changed

- **Verified:** the original pipeline reproduces byte-for-byte; 86/90 printed Report 1 values match, and the other 4 are rounding slips of 0.1.
- **Corrected:**
  - Table 10 pooled CIs now use an episode bootstrap, and the gap is +4.4 pp, not +4.5.
  - Table 12's mixed-model word-count row was a rounding and unit error.
  - The Table 11 and 13 captions are fixed.
  - The interpretation is revised: the original "role, not gender, determines" is too strong.
- **New:**
  - Measurement diagnostics.
  - Prespecified sensitivity analysis (Table S1). The gender OR ranges 0.79–1.00 depending on specification; the host OR stays between 1.50 and 2.78.
  - Leave-one-host-out check.
  - Six revised figures.
- **Human validation: not conducted and not planned.** The automated follow-up measure has **not been human-validated**. It measures **lexical uptake between adjacent turns** (whether the next turn reuses content words), and all results should be read in that sense. An annotation kit (`human_validation/`, scripts `04`–`07`, `validation_common.py`) remains in the repository **only as optional future work**. It contains no labels, is not validation, and is not required for Report 2 or the December 1 submission.

## Where things are

| What | Path |
|---|---|
| **Final LaTeX for Report 2** | `latex/topic_control_section.tex` (main), `latex/topic_control_appendix.tex` (appendix), `latex/compile_test.tex` (standalone test) |
| Scripts | `src/topic_control_revision/` (`run_all.sh` runs steps 1–4) |
| Report 1 number check | `verification/report1_number_check.csv` |
| Measurement diagnostics | `diagnostics/*.csv` |
| Corrected and new tables (CSV / Markdown / LaTeX) | `tables/` |
| Figures (PNG, 300 dpi) | `figures/` |
| Optional, unused annotation kit (future work only) | `human_validation/` |
| Pairs with turn text (rebuilt, verified against the original) | `data/mixed_pairs_with_text.csv` |
| Package versions, seeds, inputs | `run_metadata_02.json`, `human_validation/sampling_metadata.json` |

## Reproduce

```bash
PY=~/miniconda3/envs/py311/bin/python bash src/topic_control_revision/run_all.sh
```

Environment: Python 3.11.14, statsmodels 0.15.0, scipy 1.17.0, pandas 3.0.0, numpy 2.4.2, matplotlib 3.10.8. Seed 20261029. Steps 1–3 produce every result used in the reports (about 1 minute). Step 4 only redraws the optional annotation sample. Steps 5–7 are optional and are not used.

## Before committing

`data/mixed_pairs_with_text.csv` and the annotation sheets contain transcript excerpts. Transcripts themselves are git-ignored (`data/outputs/whisperx/`), so decide whether these files should go to GitHub. If the optional annotation kit is ever used, the manifest in `human_validation/manifest/` holds the automatic labels, and annotators must not open it before adjudication.
