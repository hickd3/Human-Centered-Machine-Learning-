'''
filename: viz.py
date: 12.16.24 (edits)
Authors: Dean Hickman Duilio Lucio
Purpose: Create a plot to visualize survey data

CORRECTIONS (from original):
  1. Accuracy General 4 was divided by 2 instead of 4 (line 64 in original).
     Fixed to / 4, matching the normalization in analysis_data.py.
  2. Data is now read from SurveyData.csv instead of being hard-coded inline.
     Hard-coded data is a maintenance hazard: any re-collection of data would
     require manually updating 30+ list literals.
  3. Added plt.savefig('Figure3.png', ...) before plt.show() so the figure is
     saved reproducibly without manual intervention.
  4. Docstring filename corrected (was already "viz.py", matches actual filename).
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Load & clean ──────────────────────────────────────────────────────────────
# CORRECTION 2: read from CSV rather than hard-coded dict
df = pd.read_csv('SurveyData.csv')
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how='all')

sns.set_palette("muted")

def create_visualizations(df):
    # ── Accuracy ──────────────────────────────────────────────────────────────
    df_accuracy = df[['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2',
                       'Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4']].copy()

    df_accuracy = df_accuracy.melt(
        value_vars=['Accuracy Riddle 2', 'Accuracy Logic 2', 'Accuracy General 2',
                    'Accuracy Riddle 4', 'Accuracy Logic 4', 'Accuracy General 4'],
        var_name='Accuracy Type', value_name='Accuracy'
    )

    df_accuracy['Adjusted Accuracy'] = df_accuracy['Accuracy']

    # CORRECTION 1: Accuracy General 4 divided by 4 (was / 2 in original)
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Riddle 2',   'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Logic 2',    'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy General 2',  'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 2
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Riddle 4',   'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 4
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy Logic 4',    'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 4
    df_accuracy.loc[df_accuracy['Accuracy Type'] == 'Accuracy General 4',  'Adjusted Accuracy'] = df_accuracy['Accuracy'] / 4  # FIX: was /2

    # ── Time ──────────────────────────────────────────────────────────────────
    df_time = df[['Time Riddle 2', 'Time Logic 2', 'Time General 2',
                  'Time Riddle 4', 'Time Logic 4', 'Time General 4']].melt(
        value_vars=['Time Riddle 2', 'Time Logic 2', 'Time General 2',
                    'Time Riddle 4', 'Time Logic 4', 'Time General 4'],
        var_name='Time Type', value_name='Time'
    )

    # ── Clicks ────────────────────────────────────────────────────────────────
    df_clicks = df[['Click Riddle 2', 'Click Logic 2', 'Click General 2',
                    'Click Riddle 4', 'Click Logic 4', 'Click General 4']].melt(
        value_vars=['Click Riddle 2', 'Click Logic 2', 'Click General 2',
                    'Click Riddle 4', 'Click Logic 4', 'Click General 4'],
        var_name='Click Type', value_name='Clicks'
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(x='Accuracy Type', y='Adjusted Accuracy', data=df_accuracy,
                ax=axes[0], hue='Accuracy Type', palette='Set1')
    axes[0].set_title('Adjusted Accuracy across size and group')
    axes[0].set_ylabel('Adjusted Accuracy')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(x='Time Type', y='Time', data=df_time,
                ax=axes[1], hue='Time Type', palette='Set1')
    axes[1].set_title('Time across size and group')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)

    sns.boxplot(x='Click Type', y='Clicks', data=df_clicks,
                ax=axes[2], hue='Click Type', palette='Set1')
    axes[2].set_title('Clicks across size and group')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Clicks')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # CORRECTION 3: save figure to disk before displaying
    plt.savefig('Figure3.png', dpi=150, bbox_inches='tight')
    print("Figure saved to Figure3.png")

    plt.show()

create_visualizations(df)
