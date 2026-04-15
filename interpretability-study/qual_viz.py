'''
filename: qual_viz.py
date: 12.16.24 (edits)
Authors: Dean Hickman Duilio Lucio
Purpose: Create a plot to visualize qualitative survey data

CORRECTIONS (from original):
  1. Docstring filename was "viz.py"; corrected to "qual_viz.py".
  2. Data is now read from SurveyData.csv (columns Q40_1, Q40_2, Q40_3)
     instead of being hard-coded inline, matching the approach now used
     in viz.py after its correction.
  3. Added plt.savefig('qualitative.png', ...) before plt.show() so the
     figure is saved reproducibly without manual intervention.
  4. Chart title corrected from "Average difficulty for each type"
     (shown in the rendered qualitative.png output) to "Average confidence
     for each type" to match the axis label and the Q40 survey instrument,
     which measures confidence, not difficulty.
     NOTE: the rendered qualitative.png showed "Average difficulty" as the
     title — this was a label mismatch already present in the original code;
     the original code said "Average confidence for each type" in plt.title(),
     so this correction ensures the saved file matches the code intent and
     the ylabel ("Average confidence").
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load & clean ──────────────────────────────────────────────────────────────
# CORRECTION 2: read from CSV rather than hard-coded dict
df = pd.read_csv('SurveyData.csv')
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how='all')

# Q40_1 = Riddle confidence, Q40_2 = Logic confidence, Q40_3 = General confidence
mean_riddle  = df['Q40_1'].mean()
mean_logic   = df['Q40_2'].mean()
mean_general = df['Q40_3'].mean()

categories  = ["Riddle", "Logic", "General"]
mean_values = [mean_riddle, mean_logic, mean_general]

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.bar(categories, mean_values, color=['red', 'green', 'orange'], alpha=0.7)

plt.title('Average confidence for each type', fontsize=16)   # CORRECTION 4: matches ylabel & instrument
plt.ylabel('Average confidence', fontsize=14)
plt.xlabel('Questions', fontsize=14)
plt.ylim(1, 7)

for i, value in enumerate(mean_values):
    plt.text(i, value + 0.1, f'{value:.2f}', ha='center', va='center', fontsize=12)

# CORRECTION 3: save figure to disk before displaying
plt.savefig('qualitative.png', dpi=150, bbox_inches='tight')
print("Figure saved to qualitative.png")

plt.show()
