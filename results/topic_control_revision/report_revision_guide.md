# Topic control: report revision guide (Report 2 and final capstone)

Prepared 2026-10-29. Every number below comes from files in `results/topic_control_revision/`, produced by `src/topic_control_revision/` (re-run with `bash src/topic_control_revision/run_all.sh`; outputs are identical across runs). Report 1 page numbers refer to the submitted Report 1 PDF. No editable Report 1 source or Report 2 draft was in the repository, so page references are to that PDF and no document was edited.

**Human validation: not part of this project.** No human annotation is being conducted for topic control. The automated follow-up measure has **not been human-validated**: it measures **lexical uptake between adjacent turns** (whether the next turn reuses content words), not topic follow-up as a listener would judge it. All findings below should be read in that restricted sense. The annotation materials in `human_validation/` and scripts `04`–`07` are kept in the repository **only as optional future work**. They are not completed validation and not required work.

---

## 1. Executive summary of the corrected findings

1. **The original analysis reproduces exactly.** Re-running `src/topic_control/topic_control.py` gives files byte-identical to `results/topic_control/`. 86 of 90 printed Report 1 values match; the other 4 are rounding slips of 0.1 (`verification/report1_number_check.csv`).
2. **H1 (male-initiated topics are followed up more) is not supported.** No analysis shows a male advantage. The raw pooled male advantage (+4.4 pp) reflects who is the host: 78% of male initiations come from hosts, against 30% of female initiations.
3. **The host-role association is robust.** Host-initiated topics are followed up more than guest-initiated ones in all four dyads (46–52% vs 36–41%) and in every model specification (OR 1.50–2.78). Part of this is mechanical. Questions and long replies raise word overlap, and both are more common after host turns. The association is not evidence that role *causes* uptake.
4. **The role-adjusted gender estimate depends on how the model is specified.** Within a mixed-gender episode, gender and role are perfectly linked (one male and one female speaker; one host and one guest). The "male, adjusted for role" coefficient is therefore identified only by comparing male-hosted with female-hosted episodes, which means comparing about 4 male hosts with about 4 named female hosts plus 11 unidentified ones. Across prespecified specifications the gender OR ranges from **0.79 to 1.00**: it is significant with initiator-length adjustment (A, C, E) and not significant without it (B), with reply length added (D), with host-clustered SEs (F), or in the episode-level test (G). The original OR = 0.85 (p = .002) should not be reported as a finding that male topics receive less follow-up.
5. **The measure captures lexical uptake and has not been human-validated.** In 72 of 131 mixed episodes the follow-up cutoff is 0, so "followed up" means "shares at least one content word". 89% of initiations share no content word with the previous turn, and 57% of initiations enter only through ties at the cutoff. Answers to questions are often counted as "initiations". No human validation is planned, so the report must describe the outcome as lexical uptake between adjacent turns and state the lack of validation as a limitation.
6. **The original statistical approaches answer different questions.** The paired and permutation tests estimate the gender gap *without* role adjustment. The mixed model uses a different outcome (continuous similarity). They are not failed replications of the GEE.

---

## 2. Exact changes required in Report 2

Report 2 should contain, for topic control:

| Report 2 section | What to put there | Source |
|---|---|---|
| Software development | The revision code, described as **uncommitted** until you commit it (see §13). After committing, take counts from `git log --numstat`. | `src/topic_control_revision/` |
| Experiments: topic control (revised) | §9 methodology text, §10 results text, Tables 10c, 12c, S1; Figures R1, R2, R3, R5 | §6–§10 below |
| Experiments: measurement diagnostics | Short paragraph from §10.4 plus Figure R6 | `diagnostics/` |
| Limitations | §11 text | — |
| Remaining work | §14 checklist items marked "before Dec 1" | — |

The Report 1 sentences that are now contradicted or overstated are listed in §4. Report 2 should **not** repeat any of them. Report 2 should **not** mention a human-validation study as completed, in progress or planned work. At most, add one optional future-work sentence (§11.3).

## 3. Exact changes recommended for the final capstone

- Replace the Report 1 topic-control section with the §9–§11 text (the same text as Report 2, with the final figure and table numbering).
- Define the outcome once as lexical uptake between adjacent turns and use that meaning consistently. "Topic follow-up" can remain the section name if the definition sentence is kept (§9). Do not use "topic follow-up" in the abstract or conclusions without the word "lexical" or that caveat.
- State the absence of human validation as a limitation (§11.2). No validation subsection is needed.
- Main-text figures: R1, R2, R3, R5. Appendix: R4, R6, Table 11c, Table 13c.
- No further essential analysis is required for topic control.

---

## 4. Page-by-page changes to the Report 1 topic-control section

Priority: **E** = Essential, **R** = Recommended, **O** = Optional.

### p.15, introduction paragraph 2

- **Original:** "We test whether this pattern also appears in English-language Indian podcasts featuring one host and one guest."
- **Replace with:** "We test whether this pattern also appears in Indian podcasts (English and Hindi) featuring one host and one guest; the transcripts analysed here are in English text."
- **Reason:** The dataset section (p.3) says English and Hindi. In the topic-control transcripts, Devanagari makes up 0.0002% of characters.
- **Priority:** R

- **Original:** "hosts also steer the the discussion"
- **Replace with:** "hosts also steer the discussion"
- **Priority:** E (typo)

### p.15, RQ and hypotheses

- **Keep** the RQ, H1 and H0 unchanged. The hypotheses are fine. Only the conclusions about them change.
- **Add after H0:** "Because each mixed-gender episode contains exactly one male and one female speaker, and exactly one host and one guest, speaker gender and conversational role are perfectly linked within an episode. A role-adjusted gender comparison is therefore only possible between episodes, that is, between male-hosted and female-hosted shows."
- **Priority:** E

### p.15, "Identifying initiations and follow-ups"

- **Original:** "A turn is treated as starting a new topic when it is less similar to the previous turn than the episode median. It is considered followed up when the other speaker's next turn is more similar to it than the median."
- **Replace with:** "A turn is treated as a topic initiation when its similarity to the previous turn is at or below the episode median, and the first turn of each episode is always an initiation. An initiation is counted as followed up when the other speaker's next turn is more similar to it than the episode median of these reply similarities. Because most consecutive turns share no content words, both medians are often exactly zero. The initiation cutoff is zero in 43 of the 131 mixed-gender episodes and the follow-up cutoff in 72. In these episodes, an initiation is any turn sharing no content word with the previous turn, and a follow-up is any reply sharing at least one. The measure therefore captures lexical uptake between adjacent turns, not topic structure as a human would judge it, and it has not been validated against human judgements."
- **Reason:** The code uses `<=` (`topic_control.py` line 254). `diagnostics/measurement_diagnostics.csv` shows 43/131 and 72/131 episodes with a zero cutoff and 57.2% of initiations tied at the cutoff.
- **Priority:** E

### p.15, "Statistical Analysis"

- **Original:** "We validate the result with a mixed-effects model and with paired tests comparing the male and female speakers in the same episode."
- **Replace with:** "We compare the GEE estimate with two analyses that answer related but different questions: a linear mixed model of the continuous reply similarity (a different outcome), and paired tests of the male-minus-female follow-up rate within each episode (not adjusted for role). Prespecified sensitivity analyses (Table S1) vary the turn-length adjustment, episode weighting and clustering level."
- **Reason:** "Validate" implies the same estimand. See §8.
- **Priority:** E

### p.16, "Raw comparison" paragraph

- **Original:** "After pooling across all initiations, male-initiated topics appear to be followed up more often (42.2% vs. 37.7%; Table 10). This gap appears at every threshold (Table 11). However, there is no gap Fig. 9) within the same episode."
- **Replace with:** "Pooled across all initiations, male-initiated topics are followed up more often than female-initiated topics (42.2% vs. 37.7%; +4.4 pp, episode-bootstrap 95% CI +1.9 to +6.7; Table 10). This pooled gap is present at every fixed similarity threshold (Table 11), but it is not adjusted for role. Within episodes, the male and female speaker are followed up at similar average rates (41.5% vs. 42.6%; −1.2 pp, 95% CI −4.1 to +1.8; Fig. 9)."
- **Priority:** E (fixes the broken sentence, the gap value of 4.4 not 4.5, and the role caveat)

### p.16, Table 10

- **Action:** Replace with Table 10c (§6). Pooled CIs become episode-bootstrap CIs; the gap becomes +4.4 pp.
- **Priority:** E

### p.16, Table 11

- **Action:** Correct the caption, and optionally the content (Table 11c, appendix). The original caption "unchanged by role" should read "not adjusted for role".
- **Priority:** E (caption); O (added columns)

### p.17, "Role explains the raw gap"

- **Original:** "78% of male initiations come from hosts, whereas there are only 30% of female initiations. Within each role, men and women are followed up at similar rates (Fig. 10)."
- **Replace with:** "In the mixed-gender episodes, 78% of male initiations come from hosts, compared with 30% of female initiations, so the pooled comparison mostly compares hosts with guests. Within a role, male and female initiators differ by a few percentage points (hosts: 49% for female vs. 46% for male; guests: 36% vs. 37%; Fig. 10). These within-role comparisons are between different episodes and hosts, not within the same conversation."
- **Priority:** E

### p.17–18, "Hypothesis test"

- **Original:** "The GEE results do not support H1 (Fig. 11, Table 12). After controlling for role and turn length, being the host more than doubles the odds of a topic being followed up (OR = 2.20). Male-initiated topics have slightly lower odds of follow-up (OR = 0.85, p = 0.002), which is the opposite of what H1 predicts. However, this gender effect is not supported by the mixed-effects model (p = 0.42) or the within-episode tests, with either TF-IDF or embeddings (p = 0.44 and p = 0.43; Table 13). We therefore do not reject H0."
- **Replace with:** see §10.2 (report-ready results text). The key changes:
  - The 2.20 is an *association*, not "being the host doubles the odds".
  - The 0.85 is described as sensitive to specification.
  - The mixed model and paired tests are described as different estimands, not as tests that failed to replicate the GEE.
- **Priority:** E

### p.18, Table 12

- **Action:** Replace with Table 12c (§6).
  - The mixed-model word-count row printed `0.000 [0.000, 0.000]` because the coefficient was taken from text rounded to 4 decimals and was per word, not per 100 words. The correct value is −0.08 ×10⁻² per 100 words [−0.35, 0.18], p = .53.
  - The mixed-model column header must state that its outcome is continuous similarity.
- **Priority:** E

### p.18, Table 13

- **Action:** Keep the values; they are verified. Correct the caption (§7) to state that the comparison is not adjusted for role and that the embedding measure uses its own initiation set (15,799 vs. 17,046 pairs).
- **Priority:** R

### p.18, "Role across all dyads"

- **Keep the text.** Optionally add: "This difference is consistent across dyads but partly reflects interview structure: host turns are more often questions, and questions draw replies that share their words (§10.4)."
- **Priority:** R

### p.19, Conclusion

- **Original (whole paragraph):** "Our analysis found no evidence that men's topics are followed up more than women's. The male advantage is mainly because men are usually the host in mixed-gender episodes. role, not gender, determines whose topics are taken up. This suggests that role affects whose topics are followed up more than gender. The finding also supports the idea that gendered speech patterns depend on context [6]. Without taking speaker role and episode grouping into consideration, the results could have wrongly suggested a gender bias."
- **Replace with:** see §11.1 (report-ready conclusion).
- **Reason:** "role, not gender, determines" is causal and too strong. A role-adjusted gender estimate exists, but it is inconsistent across specifications, not shown to be zero.
- **Priority:** E

### Statements now contradicted or overstated

| Report 1 statement | Status | Evidence |
|---|---|---|
| "role, not gender, determines whose topics are taken up" (p.19) | Overstated, causal | §5, Table S1 |
| "Male-initiated topics have slightly lower odds of follow-up (OR = 0.85, p = 0.002)" presented as a finding (p.17) | Contradicted as a stable finding: OR = 1.00 (p = .94) without length adjustment, 0.93 (p = .13) with reply length, p = .22 at episode level | Table S1 |
| "this gender effect is not supported by the mixed-effects model … or the within-episode tests" (p.18) | Misframed: these estimate different quantities | §8 |
| "Within each role, men and women are followed up at similar rates" (p.17) | Overstated: a 3 pp host difference, between different shows | Fig. R2 |
| "This gap appears at every threshold" (p.16), implied as robustness | Holds for the raw gap only; the within-role gaps change sign across thresholds | Table 11c |
| Table 10 pooled CIs | Too narrow (assume independent turns) | Table 10c |
| Table 12 mixed-model "+100 words 0.000 [0.000, 0.000]" | Rounding and unit error | Table 12c |
| "being the host more than doubles the odds" (p.17) | Causal wording; also 1.50–2.78 depending on adjustment | Table S1 |

---

## 5. New experiments and their findings

| # | Analysis | Finding | File |
|---|---|---|---|
| 1 | Reproduction | Byte-identical outputs; 86/90 printed values match, 4 off by 0.1 | `verification/report1_number_check.csv` |
| 2 | Threshold and tie diagnostics | Zero follow-up cutoff in 72/131 mixed episodes (133/414 overall); 57.1% of reply similarities are 0; 57.2% of initiations are ties at the cutoff; initiations make up 54.3% of turns | `diagnostics/measurement_diagnostics.csv` |
| 3 | Empty and short turns | 1,439 initiating turns and 838 replies have no content words (counted automatically as initiation and as non-follow-up, respectively); 3,574 initiations have 3 words or fewer | same |
| 4 | Question–answer structure | 40.7% of host initiations contain a question (follow-up 57.8% vs. 35.4% for non-questions); for guests 24.0% (50.0% vs. 29.8%); 34.8% of guest "initiations" directly follow a host question | `diagnostics/question_answer_structure.csv`, `initiations_that_answer_a_question.csv` |
| 5 | Length | Follow-up rises from 19–23% for 1–5-word replies to 56–61% for replies of 68+ words; female-host initiations draw median 61-word replies vs. 22 for male hosts; raw word counts are highly skewed (skewness 9.0; 0.43 after log) | `diagnostics/followup_by_length_quintile.csv`, `cell_sizes_and_lengths.csv`, `word_count_distribution.csv` |
| 6 | Host composition | M→F: 4 hosts (Raj Shamani 32 episodes, Ranveer 26, Prakhar Gupta 5, Samdish Bhatia 4). F→M: 4 named hosts (Karishma Mehta 28, Faye Dsouza 19, Pooja Dhingra 5, Soha Ali Khan 1) plus 11 episodes whose host could not be linked to a known speaker | `diagnostics/host_composition.csv` |
| 7 | Sensitivity analysis A–G | Gender OR 0.79–1.00, significance depends on specification; host OR 1.50–2.78, always significant | `tables/table_S1_sensitivity.*`, `tables/sensitivity_full.csv` |
| 8 | Leave-one-host-out (A, C) | Dropping any one host with ≥4 episodes leaves spec A's OR at 0.79–0.88 (all p < .05): no single host drives the specification-A result | `tables/leave_one_host_out.csv` |
| 9 | Embedding comparability | Embedding check uses a different initiation set (15,799 vs. 17,046 mixed pairs) and only role-unadjusted paired tests | `diagnostics/embedding_check_comparability.csv` |

---

## 6. Corrected tables (paths and recommended placement)

All tables exist as `.csv`, `.md` and `.tex` in `results/topic_control_revision/tables/`.

| Report 1 | Action | New file | Placement |
|---|---|---|---|
| Table 10 | **Replace** | `table10_corrected` | Main |
| Table 11 | **Correct caption**; optional replacement | `table11_corrected` | Appendix |
| Table 12 | **Replace** | `table12_corrected` | Main |
| Table 13 | **Keep values, fix caption** | `table13_corrected` | Appendix |
| — | **Add** | `table_S1_sensitivity` | Main |

### Table 10c (replaces Table 10)

| Initiator | Initiations | Pooled rate [episode-bootstrap 95% CI] | Per-episode mean [95% CI] | Per-episode median |
|---|---|---|---|---|
| Female | 8,304 | 37.7% [35.7, 40.0] | 42.6% [40.6, 44.7] | 43.3% |
| Male | 8,742 | 42.2% [40.5, 43.9] | 41.5% [39.8, 43.2] | 40.6% |
| Male − female | | +4.4 pp [+1.9, +6.7] | −1.2 pp [−4.1, +1.8] | +1.2 pp* |

*Median of the within-episode differences. Note that this is not the difference of the two medians.

### Table 12c (replaces Table 12)

| Predictor | GEE OR [95% CI] | GEE p | Mixed model β ×100 [95% CI] | Mixed model p |
|---|---|---|---|---|
| Male initiator (vs female) | 0.85 [0.76, 0.94] | .002 | 0.21 [−0.30, 0.72] | .422 |
| Host initiator (vs guest) | 2.20 [1.99, 2.44] | < .001 | 1.85 [1.33, 2.38] | < .001 |
| Initiator turn length, per 100 words | 2.64 [2.26, 3.08] | < .001 | −0.08 [−0.35, 0.18] | .533 |

### Table S1 (new)

| Spec | Model | Unit | Male initiator OR [95% CI] | p | Host initiator OR [95% CI] |
|---|---|---|---|---|---|
| A | Original (Report 1) | Pair, episode clusters | 0.85 [0.76, 0.94] | .002 | 2.20 [1.99, 2.44] |
| B | No length adjustment | Pair, episode clusters | 1.00 [0.91, 1.09] | .939 | 1.50 [1.37, 1.64] |
| C | Log initiator length | Pair, episode clusters | 0.79 [0.72, 0.87] | < .001 | 2.78 [2.53, 3.06] |
| D | Log initiator + reply length | Pair, episode clusters | 0.93 [0.85, 1.02] | .126 | 1.85 [1.69, 2.03] |
| E | Equal episode weights | Pair, episode clusters | 0.83 [0.75, 0.92] | < .001 | 2.22 [1.99, 2.47] |
| F | Host-clustered SE† | Pair, 19 host clusters | 0.85 [0.70, 1.02] | .083 | 2.21 [1.57, 3.12] |
| G | Episode-level gap comparison | Episode (n = 131) | Host − guest gap: M→F 9.4 pp vs F→M 12.3 pp; difference −2.8 pp [−7.4, +1.6] | .222 | — |

†Reported for transparency only: 4 male-host clusters are too few for reliable cluster-robust inference.

Table 11c (appendix) is in `tables/table11_corrected.md`. Its key added information: the within-role gaps change sign across thresholds. The host-initiated gap goes from −4.3 pp at 0.02 to +3.5 pp at 0.10, and the guest-initiated gap from +2.6 pp to −4.0 pp.

## 7. Table and figure captions

The final, authoritative captions are in **`latex/topic_control_section.tex`** (main report) and **`latex/topic_control_appendix.tex`** (appendix). They are not repeated here, so there is only one version of each caption.

| Report 2 item | Label | Revised file | Placement | Replaces (Report 1) |
|---|---|---|---|---|
| Table: follow-up by gender, pooled and per episode | `tab:tc-descriptive` | `tables/table10_corrected.*` | Main | Table 10 |
| Table: GEE and mixed-model predictors | `tab:tc-models` | `tables/table12_corrected.*` | Main | Table 12 |
| Table: sensitivity A–G | `tab:tc-sensitivity` | `tables/table_S1_sensitivity.*` | Main | new |
| Table: fixed thresholds | `tab:tc-thresholds` | Report 1 values (verified), caption corrected | Appendix | Table 11 |
| Table: paired within-episode, TF-IDF vs embeddings | `tab:tc-within` | `tables/table13_corrected.*` | Appendix | Table 13 |
| Fig. R1 host vs guest by dyad | `fig:tc-dyad` | `figures/fig1_followup_by_role_all_dyads.png` | Main | Fig. 12 |
| Fig. R2 gender within role | `fig:tc-role` | `figures/fig2_followup_by_gender_within_role.png` | Main | Fig. 10 |
| Fig. R3 within-episode differences | `fig:tc-episode` | `figures/fig3_within_episode_gender_difference.png` | Main | Fig. 9 |
| Fig. R5 sensitivity | `fig:tc-sensitivity` | `figures/fig5_gender_estimate_sensitivity.png` | Main | new |
| Fig. R4 original GEE | `fig:tc-gee` | `figures/fig4_gee_original_model.png` | Appendix | Fig. 11 |
| Fig. R6 follow-up by reply length | `fig:tc-length` | `figures/fig6_followup_by_reply_length.png` | Appendix | new |

**Verification notes (final pass):**
- *Table 13, embedding row:* recomputed from `results/topic_control/topic_control_embeddings_turns.csv` and confirmed: female 48.37%, male 47.20%, gap −1.18 pp [−4.15, +1.79], paired t = −0.784 (p = .435), permutation p = .441, 15,799 initiations. The values resemble the TF-IDF row (−1.16 pp [−4.09, +1.77], t = −0.782, p = .436) but are independent: the per-episode gaps under the two measures correlate at r = 0.70.
- *Table 12 units:* both models use initiating-turn words ÷ 100. The GEE odds ratio per 100 words is 2.64 [2.26, 3.08] (log-odds 0.971, SE 0.080). The mixed-model coefficient per 100 words is −0.00084 change in cosine similarity [−0.0035, +0.0018], p = .53, shown ×100 as −0.08 [−0.35, 0.18]. Report 1's `0.000 [0.000, 0.000]` came from rounding a per-word coefficient to 4 decimals.

## 8. Why the original statistical results differ

- **Paired and permutation tests** (Table 13) estimate the average within-episode male-minus-female difference **without adjusting for role**. In male-hosted episodes this difference equals the host–guest gap (+9.4 pp). In female-hosted episodes it equals minus that gap (−12.3 pp). The overall −1.2 pp is the average of the two, so these tests ask whether gender matters when role is ignored. They do not test the GEE's coefficient.
- **The GEE gender coefficient** is adjusted for role. Since each episode has one speaker of each gender and one of each role, the coefficient equals half the difference between the host–guest gap in male-hosted and in female-hosted episodes. It is estimated entirely *between* shows, from 4 male hosts, 4 named female hosts and 11 unidentified hosts. The episode-level version of exactly this comparison (spec G) is −2.8 pp, 95% CI −7.4 to +1.6, p = .22. The turn-level GEE adds covariate adjustment; its significance comes from the initiator-length term, which is highly skewed and differs greatly between cells (male guests in female-hosted episodes average 82 words; median 34). Without length adjustment OR = 1.00. With log length OR = 0.79. Adding reply length gives OR = 0.93.
- **The mixed model** analyses a different outcome: the continuous similarity, not whether it crosses the episode median. 57% of values are exactly 0 and the residuals are highly skewed (skewness 4.7), so a normal linear mixed model is a poor fit, and its p-values should not be used to confirm or reject the GEE. The variational-Bayes random-intercept logistic model in the original script (same binary outcome) gives a gender OR of 0.83 in the same direction. That script itself notes its uncertainty is unreliable.
- **Conclusion:** none of the three approaches gives a stable, interpretable gender effect, and each answers a narrower question than the report text implied.

## 9. Report-ready methodology text

*The final wording is in `latex/topic_control_section.tex`. §9–§11 below are the longer draft wording, and the numbers are identical. Where the two differ, use the LaTeX files.*

> **Data.** The analysis uses the 131 mixed-gender episodes (67 male host→female guest, 64 female host→male guest) of the 418-episode corpus, with descriptive comparisons across all 414 episodes with transcripts. Consecutive lines from the same speaker are merged into turns, and turns consisting only of a backchannel (e.g. "yeah", "mhm") are removed. Other short turns are kept. This leaves 31,393 turns in the mixed-gender episodes.
>
> **Topic initiation and follow-up.** Each turn is represented as a TF-IDF vector of non-stopword unigrams and bigrams, computed within each episode. A turn is treated as a topic initiation when its cosine similarity to the previous turn is at or below the episode median; the first turn of each episode is always an initiation. The response is the other speaker's next turn. The initiation is counted as followed up when the response's similarity to it exceeds the episode median of these response similarities. This yields 17,046 initiation–response pairs (8,742 male-initiated, 8,304 female-initiated). Because most consecutive turns share no content words, the follow-up cutoff is zero in 72 of the 131 episodes. In those episodes, "followed up" means the reply shares at least one content word with the initiating turn. The measure therefore captures lexical uptake between adjacent turns. It has not been compared with human judgements of topic change or engagement, and throughout this section "follow-up" is used only in this lexical sense.
>
> **Statistical analysis.** Turns within an episode are not independent, and within each mixed-gender episode gender and role are perfectly linked (one male and one female speaker; one host and one guest). The primary model is a GEE logistic regression of follow-up on initiator gender, initiator role and initiator turn length, with an exchangeable working correlation and robust standard errors clustered by episode (Report 1). Because the gender coefficient in this model is identified only by comparing male-hosted with female-hosted episodes, we prespecified six alternative analyses (Table S1):
> - (B) no length adjustment;
> - (C) log-transformed initiator length;
> - (D) log initiator and log reply length;
> - (E) equal weight per episode;
> - (F) standard errors clustered by host;
> - (G) a direct episode-level comparison of the host-minus-guest follow-up gap between male-hosted and female-hosted episodes.
>
> Specification D conditions on the length of the reply, which is part of the response being measured, so it estimates the association with gender and role *other than* through reply length. Specification F is shown for transparency only, because only four male hosts form clusters. We also report the within-episode male-minus-female difference (paired t and sign-flip permutation tests; not adjusted for role) and, for comparison with Report 1, a linear mixed model of the continuous reply similarity (a different outcome). Pooled rates are given with 95% confidence intervals from 5,000 bootstrap resamples of episodes.

## 10. Report-ready results text

### 10.1 Raw comparison

> Pooled across all initiations, male-initiated topics were followed up more often than female-initiated topics (42.2% vs. 37.7%; +4.4 pp, 95% CI +1.9 to +6.7; Table 10). However, 78% of male initiations came from hosts, compared with 30% of female initiations. Within episodes, the male and female speaker were followed up at similar average rates (41.5% vs. 42.6%; −1.2 pp, 95% CI −4.1 to +1.8; paired t = −0.78, p = .44; Table 13). The within-episode difference follows role: it is positive when the man is the host (+9.4 pp) and negative when the woman is the host (−12.3 pp; Fig. R3).

### 10.2 Hypothesis test and sensitivity

> No analysis supported H1. In the original GEE model, host-initiated topics had higher odds of follow-up than guest-initiated topics (OR = 2.20, 95% CI 1.99–2.44), and male-initiated topics had lower odds than female-initiated topics (OR = 0.85, 0.76–0.94, p = .002; Table 12). The role association held in every prespecified specification (OR 1.50–2.78; Table S1, Fig. R5). The gender estimate did not: it was 1.00 (p = .94) without length adjustment, 0.79 (p < .001) with log initiator length, 0.93 (p = .13) when reply length was also included, and 0.85 (p = .08) with host-clustered standard errors. The episode-level comparison that underlies the role-adjusted gender estimate found no reliable difference between male-hosted and female-hosted episodes in the host–guest gap (9.4 vs. 12.3 pp; difference −2.8 pp, 95% CI −7.4 to +1.6, p = .22). Removing any single host left the original estimate between 0.79 and 0.88. The negative gender estimate is therefore not driven by one show, but its size and significance depend on how turn length is modelled. We therefore do not reject H0, and we do not interpret the original OR = 0.85 as evidence that men's topics are followed up less.

### 10.3 All dyads

> Across all 414 episodes (42,320 initiations), host-initiated topics were followed up more often than guest-initiated topics in every dyad (46–52% vs. 36–41%; Fig. R1), whether the host was male or female.

### 10.4 Measurement diagnostics

> The automatic measure is closely tied to word overlap. Turns containing a question were followed up far more often than other turns (hosts 57.8% vs. 35.4%; guests 50.0% vs. 29.8%), and 41% of host initiations but 24% of guest initiations contained a question. Follow-up also rose steeply with the length of the reply, from about 20% for replies of 1–5 words to about 60% for replies of 68 words or more (Fig. R6). Replies to female hosts were much longer than replies to male hosts (median 61 vs. 22 words). Turns with no content words (1,439 initiations, 838 replies) are classified automatically, as initiations and as non-follow-ups respectively. About a third of guest "initiations" are direct responses to a host question. Part of the host advantage therefore reflects the question–answer structure of interviews and not topic uptake as such.

## 11. Report-ready discussion and limitations

### 11.1 Conclusion (replaces Report 1 p.19 Conclusion)

> Using a lexical measure of uptake between adjacent turns, we found no evidence that topics introduced by men are followed up more than topics introduced by women in mixed-gender podcast conversations. The male advantage in pooled follow-up rates disappears within episodes, because in this corpus men are usually the host and hosts' topics are taken up more often. The association between the host role and follow-up is consistent across dyads and model specifications. However, it partly reflects interview structure (questions and long answers that repeat the question's words), and it is an association, not evidence that role causes uptake. The absence of support for H1 is not evidence that gender plays no role. A role-adjusted gender comparison is possible only between shows, rests on a small number of hosts, and gives estimates that change with how turn length is modelled. These results are consistent with the view that gendered conversational patterns depend on context [6], and they show that analyses that ignore role and episode structure can produce a misleading gender gap.

### 11.2 Limitations

> - **Measurement.** Topic initiation and follow-up are inferred from word overlap between adjacent turns. In 72 of the 131 mixed-gender episodes, "followed up" reduces to sharing a single content word, and 57% of initiations enter the sample through ties at the cutoff. Paraphrased uptake is missed, and lexical repetition without engagement is counted. The embedding-based check uses a different initiation set and tests only the role-unadjusted gap, so it does not address this. The measure has not been compared with human judgements. The results therefore describe lexical uptake between adjacent turns, not topic follow-up as a listener would perceive it, and the error rate of the automatic labels is unknown.
> - **Role and gender are linked.** In mixed-gender host–guest episodes, gender and role cannot be separated within an episode. Any role-adjusted gender effect is a comparison between male-hosted and female-hosted shows.
> - **Few hosts.** The mixed-gender sample contains 4 male hosts (two of whom account for 58 of the 67 male-hosted episodes) and 4 named female hosts (two account for 47 of 64), plus 11 female-hosted episodes whose host could not be linked to a known speaker. Results may reflect particular shows and formats more than gender.
> - **Turn length.** Turn lengths are highly skewed, and both initiation length and reply length strongly predict measured follow-up. Reply length is partly a consequence of the initiation, so adjusting for it changes what is estimated.
> - **Scope.** Only the immediately following turn is examined; uptake later in the conversation is not counted. Very short non-backchannel turns are retained.
> - **Generalisability.** Associations only; the corpus is a convenience sample of popular Indian YouTube podcasts.

### 11.3 Future work

> Future work could test the automatic labels against human judgements of topic change and engagement, use a multi-turn uptake window, model host identity explicitly as a random effect, and use a topic segmentation method that does not depend on shared words.

---

## 12. Human validation: optional future work only (not conducted, not required)

No human annotation is being conducted for topic control, and none is required for Report 2 or the December 1 submission. These materials remain in the repository unused, in case anyone wants them later:

- `human_validation/`: a 176-pair stratified sample with recorded inclusion probabilities, blank blinded sheets, `ANNOTATION_GUIDE.md`, and an adjudication template;
- `src/topic_control_revision/04_prepare_validation.py`–`07_validation_error_by_group.py` and `validation_common.py`.

They contain **no labels and no results**. Do not describe them in either report as validation, and do not list them as remaining work. If they are ever used, follow `ANNOTATION_GUIDE.md` and report design-weighted estimates only.

## 13. Software development (for Report 2)

- **No commits.** The topic-control revision has **not been committed**. `git status` shows `src/topic_control_revision/` and `results/topic_control_revision/` as untracked, so there are no commit statistics for this work yet.
- **New code:** 10 files in `src/topic_control_revision/` (9 Python scripts and modules plus `run_all.sh`), 1,593 lines in total by `wc -l`.
- **No existing files modified.** The original topic-control scripts and outputs were not changed.
- **To report commit statistics, commit first.** Commit the work (and decide whether the annotation sheets, which contain transcript excerpts, should go to GitHub; transcripts themselves are git-ignored). Then take the counts from `git log --numstat`.
- **Of these, the analysis code is 5 files:** `common.py`, `01`–`03` and `run_all.sh`, totalling 1,107 lines. The other 486 lines (`04`–`07`, `validation_common.py`) are the unused, optional validation tooling.
- **Description for Report 2:** "Added a reproducible revision pipeline for the topic-control analysis: verification of all Report 1 values against saved outputs, measurement diagnostics, prespecified GEE sensitivity analyses with episode- and host-level uncertainty, corrected tables with episode-bootstrap confidence intervals, and revised figures."

## 14. Prioritised checklist

### Report 2: essential

- [ ] **Paste the final LaTeX:** `latex/topic_control_section.tex` replaces the Report 1 topic-control section, and `latex/topic_control_appendix.tex` goes in an appendix of the same document so the cross-references resolve. Copy the six PNGs into your report's `figures/` folder.

- [ ] **Methodology:** paste §9. It defines follow-up as lexical uptake and states that the measure is not human-validated.
- [ ] **p.15 fixes:** the initiation rule (§4, "Identifying initiations"), the "Statistical Analysis" replacement (§4), the "the the" typo, and the gender/role sentence after H0 (§4).
- [ ] **Results:** paste §10.1, §10.2 and §10.3. Paste §10.4 too (measurement diagnostics); it supports the lexical-uptake caveat.
- [ ] **Conclusion:** replace the Report 1 p.19 conclusion with §11.1.
- [ ] **Limitations:** paste §11.2. It includes the statement that the measure has not been human-validated.
- [ ] **Tables:** Table 10c (`tables/table10_corrected.*`), Table 12c (`tables/table12_corrected.*`) and Table S1 (`tables/table_S1_sensitivity.*`) in the main text, with the §7 captions. Correct the Table 11 and 13 captions, or move 11c and 13c to the appendix.
- [ ] **Main figures, with the §7 captions:**
  - `figures/fig1_followup_by_role_all_dyads.png` (R1)
  - `figures/fig2_followup_by_gender_within_role.png` (R2)
  - `figures/fig3_within_episode_gender_difference.png` (R3)
  - `figures/fig5_gender_estimate_sensitivity.png` (R5)
- [ ] **Appendix figures:** `figures/fig4_gee_original_model.png` (R4) and `figures/fig6_followup_by_reply_length.png` (R6).
- [ ] **Remove the contradicted Report 1 statements** listed in §4.
- [ ] **Software section:** describe the work as in §13. Count it as commits only after you commit it, using `git log --numstat`.
- [ ] **Do not mention a human-validation study** as completed, ongoing or remaining work.

### Before December 1

- [ ] Carry the same text, tables and figures into the final capstone with final numbering (§3).
- [ ] Check that the abstract and overall conclusions describe topic control as lexical uptake and do not claim a gender effect or a causal role effect.

### Optional (not required)

- [ ] Human validation with the unused kit (§12); multi-turn uptake window; host random-effects model; restricting to non-question initiations; multilingual embeddings.
