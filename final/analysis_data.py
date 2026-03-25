'''
filename: analysis_data.py
date: 12.16.24 (edits)
Authors: Dean Hickman
Purpose: Conduct statistical tests for analyzing survey data
'''

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# Load data from CSV
df = pd.read_csv('SurveyData.csv')

#clean up

#white space
df.columns = df.columns.str.strip()

# Drop fully empty columns
df = df.dropna(axis=1, how='all')

# Normalize accuracy scores
df['Accuracy Riddle 2'] = df['Accuracy Riddle 2'] / 2
df['Accuracy Logic 2']  = df['Accuracy Logic 2']  / 2
df['Accuracy General 2'] = df['Accuracy General 2'] / 2
df['Accuracy Riddle 4'] = df['Accuracy Riddle 4'] / 4
df['Accuracy Logic 4']  = df['Accuracy Logic 4']  / 4
df['Accuracy General 4'] = df['Accuracy General 4'] / 4


# --- Paired t-tests ---

# Riddle
t_stat, p_val = stats.ttest_rel(df['Accuracy Riddle 2'], df['Accuracy Riddle 4'])
print(f"Paired t-test for Accuracy Riddle:  t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Time Riddle 2'], df['Time Riddle 4'])
print(f"Paired t-test for Time Riddle:      t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Click Riddle 2'], df['Click Riddle 4'])
print(f"Paired t-test for Click Riddle:     t = {t_stat:.3f}, p = {p_val:.3f}")

# Logic
t_stat, p_val = stats.ttest_rel(df['Accuracy Logic 2'], df['Accuracy Logic 4'])
print(f"Paired t-test for Accuracy Logic:   t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Time Logic 2'], df['Time Logic 4'])
print(f"Paired t-test for Time Logic:       t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Click Logic 2'], df['Click Logic 4'])
print(f"Paired t-test for Click Logic:      t = {t_stat:.3f}, p = {p_val:.3f}")

# General
t_stat, p_val = stats.ttest_rel(df['Accuracy General 2'], df['Accuracy General 4'])
print(f"Paired t-test for Accuracy General: t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Time General 2'], df['Time General 4'])
print(f"Paired t-test for Time General:     t = {t_stat:.3f}, p = {p_val:.3f}")

t_stat, p_val = stats.ttest_rel(df['Click General 2'], df['Click General 4'])
print(f"Paired t-test for Click General:    t = {t_stat:.3f}, p = {p_val:.3f}")


# --- ANOVA: Accuracy by question type within each block size ---

anova_data_2 = pd.melt(
    df[['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2']],
    var_name='Type', value_name='Accuracy'
)
model_2 = ols('Accuracy ~ C(Type)', data=anova_data_2).fit()
print("\nANOVA result for Accuracy (2-question block):")
print(anova_lm(model_2))

anova_data_4 = pd.melt(
    df[['Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4']],
    var_name='Type', value_name='Accuracy'
)
model_4 = ols('Accuracy ~ C(Type)', data=anova_data_4).fit()
print("\nANOVA result for Accuracy (4-question block):")
print(anova_lm(model_4))


# --- ANOVA: Effect of block size across all question types ---

# Accuracy
anova_accuracy = pd.melt(
    df[['Accuracy Riddle 2', 'Accuracy Riddle 4',
        'Accuracy Logic 2',  'Accuracy Logic 4',
        'Accuracy General 2', 'Accuracy General 4']],
    var_name='Block', value_name='Accuracy'
)
anova_accuracy['Block_Size'] = anova_accuracy['Block'].apply(lambda x: 2 if '2' in x else 4)
accuracy_model = ols('Accuracy ~ Block_Size', data=anova_accuracy).fit()
print("\nANOVA result for Accuracy (block size effect):")
print(anova_lm(accuracy_model))

# Time
anova_time = pd.melt(
    df[['Time Riddle 2', 'Time Riddle 4',
        'Time Logic 2',  'Time Logic 4',
        'Time General 2', 'Time General 4']],
    var_name='Block', value_name='Time'
)
anova_time['Block_Size'] = anova_time['Block'].apply(lambda x: 2 if '2' in x else 4)
time_model = ols('Time ~ Block_Size', data=anova_time).fit()
print("\nANOVA result for Time (block size effect):")
print(anova_lm(time_model))

# Clicks
anova_clicks = pd.melt(
    df[['Click Riddle 2', 'Click Riddle 4',
        'Click Logic 2',  'Click Logic 4',
        'Click General 2', 'Click General 4']],
    var_name='Block', value_name='Clicks'
)
anova_clicks['Block_Size'] = anova_clicks['Block'].apply(lambda x: 2 if '2' in x else 4)
click_model = ols('Clicks ~ Block_Size', data=anova_clicks).fit()
print("\nANOVA result for Clicks (block size effect):")
print(anova_lm(click_model))
