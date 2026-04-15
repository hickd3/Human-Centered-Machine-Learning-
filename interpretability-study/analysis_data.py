'''
filename: analysis_data.py
date: 12.16.24 (edits)
Authors: Dean Hickman
Purpose: Conduct statistical tests for analyzing survey data

CORRECTIONS (from original):
  1. Added Bonferroni correction across the 9 paired t-tests (alpha=0.05 -> alpha_corrected=0.0056).
     Corrected alpha is printed before the t-test block and flagged on each result line.
  2. Added Cohen's d effect size for every paired t-test.
  3. viz.py hard-codes data inline and does not read this CSV, so the
     normalization here was always correct (General 4 / 4). No change needed.
  4. Added plt-free savefig-equivalent: results are also written to
     analysis_results.txt so the output is reproducible headlessly.
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('SurveyData.csv')

# ── Clean ─────────────────────────────────────────────────────────────────────
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how='all')   # removes phantom blank column in raw CSV

# ── Normalize accuracy to [0, 1] proportions ─────────────────────────────────
df['Accuracy Riddle 2']  = df['Accuracy Riddle 2']  / 2
df['Accuracy Logic 2']   = df['Accuracy Logic 2']   / 2
df['Accuracy General 2'] = df['Accuracy General 2'] / 2
df['Accuracy Riddle 4']  = df['Accuracy Riddle 4']  / 4
df['Accuracy Logic 4']   = df['Accuracy Logic 4']   / 4
df['Accuracy General 4'] = df['Accuracy General 4'] / 4   # FIX: was /2 in viz.py; correct value is /4

# ── Helper: Cohen's d for paired samples ─────────────────────────────────────
def cohens_d_paired(a, b):
    """Cohen's d = mean(diff) / std(diff) for paired observations."""
    diff = a - b
    return diff.mean() / diff.std(ddof=1)

# ── Bonferroni correction ─────────────────────────────────────────────────────
# CORRECTION 1: 9 paired t-tests are conducted; without correction the
# familywise error rate is inflated. Bonferroni divides alpha by the number
# of comparisons in the family.
N_TESTS   = 9
ALPHA     = 0.05
ALPHA_CORR = ALPHA / N_TESTS   # 0.00556

lines = []   # collect output for file save

header = (
    f"\n{'='*70}\n"
    f"Bonferroni-corrected alpha = {ALPHA} / {N_TESTS} = {ALPHA_CORR:.5f}\n"
    f"Significance marker (*) indicates p < {ALPHA_CORR:.5f}\n"
    f"{'='*70}\n"
)
print(header)
lines.append(header)

# ── Paired t-tests ────────────────────────────────────────────────────────────
def run_ttest(col_a, col_b, label):
    t, p = stats.ttest_rel(df[col_a], df[col_b])
    d    = cohens_d_paired(df[col_a], df[col_b])
    sig  = "*" if p < ALPHA_CORR else " "
    line = (f"{sig} {label:<40} t = {t:7.3f},  p = {p:.4f},  d = {d:.3f}")
    print(line)
    lines.append(line)

print("--- Paired t-tests (2-block vs 4-block) ---")
lines.append("--- Paired t-tests (2-block vs 4-block) ---")

# Riddle
run_ttest('Accuracy Riddle 2',  'Accuracy Riddle 4',  'Accuracy Riddle:')
run_ttest('Time Riddle 2',      'Time Riddle 4',      'Time Riddle:')
run_ttest('Click Riddle 2',     'Click Riddle 4',     'Click Riddle:')

# Logic
run_ttest('Accuracy Logic 2',   'Accuracy Logic 4',   'Accuracy Logic:')
run_ttest('Time Logic 2',       'Time Logic 4',       'Time Logic:')
run_ttest('Click Logic 2',      'Click Logic 4',      'Click Logic:')

# General
run_ttest('Accuracy General 2', 'Accuracy General 4', 'Accuracy General:')
run_ttest('Time General 2',     'Time General 4',     'Time General:')
run_ttest('Click General 2',    'Click General 4',    'Click General:')

# ── ANOVA: Accuracy by question type within each block size ───────────────────
def run_anova(cols, label):
    melted = pd.melt(df[cols], var_name='Type', value_name='Accuracy')
    model  = ols('Accuracy ~ C(Type)', data=melted).fit()
    result = anova_lm(model)
    header = f"\nANOVA — {label}:"
    print(header)
    print(result.to_string())
    lines.append(header)
    lines.append(result.to_string())
    return model

run_anova(
    ['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2'],
    'Accuracy by question type (2-question block)'
)
run_anova(
    ['Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4'],
    'Accuracy by question type (4-question block)'
)

# ── ANOVA: Effect of block size across all question types ─────────────────────
def run_block_anova(wide_cols, outcome_name):
    melted = pd.melt(df[wide_cols], var_name='Block', value_name=outcome_name)
    melted['Block_Size'] = melted['Block'].apply(lambda x: 2 if '2' in x else 4)
    model  = ols(f'{outcome_name} ~ Block_Size', data=melted).fit()
    result = anova_lm(model)
    header = f"\nANOVA — {outcome_name} (block size effect):"
    print(header)
    print(result.to_string())
    lines.append(header)
    lines.append(result.to_string())

run_block_anova(
    ['Accuracy Riddle 2', 'Accuracy Riddle 4',
     'Accuracy Logic 2',  'Accuracy Logic 4',
     'Accuracy General 2','Accuracy General 4'],
    'Accuracy'
)
run_block_anova(
    ['Time Riddle 2', 'Time Riddle 4',
     'Time Logic 2',  'Time Logic 4',
     'Time General 2','Time General 4'],
    'Time'
)
run_block_anova(
    ['Click Riddle 2', 'Click Riddle 4',
     'Click Logic 2',  'Click Logic 4',
     'Click General 2','Click General 4'],
    'Clicks'
)

# ── Save results to file ──────────────────────────────────────────────────────
# CORRECTION 4: original scripts had no file output; results were lost after
# each interactive run. Write a plain-text record alongside the script.
with open('analysis_results.txt', 'w') as f:
    f.write('\n'.join(lines))
print("\nResults written to analysis_results.txt")
