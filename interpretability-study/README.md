# Decision Tree Interpretability via Proxy Survey
---

## Overview

This project investigates how interpretability in decision tree models changes as a function of **path length** (tree depth) and **question type** (context clarity). Rather than studying a live ML system, we use a survey-based proxy where three question categories simulate distinct model opacity levels:

| Question Type | Decision Tree Analog | Rationale |
|---|---|---|
| **Riddle** | Black-box model | Intuitive-but-wrong first answers; hint required after failure |
| **Logic** | White-box (known rules) | All information provided; interpretability tested via reasoning load |
| **General Knowledge** | White-box (innate knowledge) | Minimal cognitive load; user either knows or doesn't |

Participants completed both a **2-question block** (shorter path) and a **4-question block** (longer path) for each type in a within-subjects design (N = 16).

Full write-ups can be provided upon request. The survey is currently unavailable due to the author no longer having access permissions to the hosting platform.
---

## Research Questions

1. Does path length (block size) affect accuracy on decision-tree-proxy tasks?
2. Does path length affect cognitive load, as measured by time-on-task and number of clicks?
3. Does question type (model opacity analog) modulate these effects?

---

## Key Findings

**Accuracy is unaffected by path length or question type.** No significant differences were found across block sizes or question types in any accuracy analysis (all p > .18 after Bonferroni correction).

**Path length significantly increases time and clicks.** Across all question types, the 4-block condition produced substantially more time-on-task (F(1, 94) = 34.26, p < .001) and more clicks (F(1, 94) = 24.44, p < .001), with large effect sizes.

| Measure | Block Size Effect | Representative Cohen's d |
|---|---|---|
| Accuracy | n.s. (p = .640) | — |
| Time | p < .001 | −0.98 to −2.00 |
| Clicks | p < .001 | −0.90 to −1.65 |

**Riddle questions carry the highest cognitive load.** They produced the largest click counts and the lowest self-reported confidence (mean = 3.62 on a 1–7 Likert scale, vs. 5.62 for Logic and 5.50 for General). Qualitative feedback corroborated this: the majority of participants named Riddle questions as the most difficult type.

**Implication:** Interpretability cannot be evaluated on accuracy alone. Users may answer correctly under longer/more opaque decision paths but at measurable cognitive cost. Time and interaction count should be included as interpretability metrics in human-in-the-loop decision tree systems.

---

## Repository Structure

```
.
├── SurveyData.csv          # Raw per-participant survey responses (N=16)
├── clean_data.csv          # Cleaned version of survey data
├── analysis_data.py        # Statistical analysis: paired t-tests + ANOVAs
├── analysis_results.txt    # Saved output from analysis_data.py
├── viz.py                  # Boxplot visualizations (accuracy, time, clicks)
├── qual_viz.py             # Bar chart of self-reported confidence by question type
├── Figure2.png             # [Study design / descriptive figure]
├── Figure3.png             # Boxplots: adjusted accuracy, time, clicks by condition
├── qualitative.png         # Bar chart: average confidence per question type
└── Interpretability-study_writeup.pdf  # Full write-up
```

---

## Methods Summary

- **Design:** Within-subjects; each participant completed both block sizes for all three question types
- **N:** 16 participants
- **Metrics:** Accuracy (proportion correct), time to completion (seconds), clicks per block, self-reported confidence (Q40, 1–7 Likert)
- **Statistics:** 9 paired t-tests with Bonferroni correction (α* = .006); Cohen's d for all comparisons; one-way ANOVAs within and across block sizes
- **Survey platform:** Qualtrics

---

## Limitations

- **Small sample (N = 16):** Limits statistical power and generalizability; all effects should be interpreted cautiously
- **One extreme outlier retained:** One participant's session duration was ~18.7 hours (67,220 s); no sensitivity analysis was conducted excluding this case
- **Two path-length levels only:** The ANOVA treating block size as a continuous predictor has only two levels; results cannot speak to nonlinear depth effects
- **Unvalidated proxy instrument:** The mapping from question ambiguity to model opacity is theoretical and has not been independently validated
- **Counterbalancing unclear:** Ordering effects across blocks and question types may not be fully controlled

---

## Potential Extensions

- Scale to N ≥ 60 for adequate power; pre-register exclusion criteria for outlier sessions
- Add intermediate path lengths (e.g., 6- and 8-question blocks) to model depth effects continuously
- Validate the question-type–to–model-type analogy against a real decision tree system
- Collect per-block confidence ratings rather than a single end-of-survey rating
- Incorporate eye-tracking or mouse-movement data as additional cognitive load proxies

---

## Dependencies

```
Python 3.x
pandas, numpy, scipy, statsmodels, matplotlib, seaborn
```

Run analysis:
```bash
python analysis_data.py   # writes results to analysis_results.txt
python viz.py             # saves Figure3.png
python qual_viz.py        # saves qualitative.png
```

All scripts read from `SurveyData.csv` in the working directory.

---

## Contributions
 Duilio Lucio, co-PI
