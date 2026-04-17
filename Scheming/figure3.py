'''
filename: figure3.py
Authors: Dean Hickman, Duilio Lucio
Purpose: Bar chart of observed scheming rates across the three evaluated frontier models.
         Data is simulated from evaluation observation frequencies (GPT o1*: 8/10,
         Gemini 2.0: 0/10, Claude 3.7: 0/5). This plot corresponds to Figure 3 in
         the paper.

'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Simulated trial-level data from evaluation observation frequencies
# GPT o1*: 8/10 trials exhibited scheming; Gemini 2.0 and Claude 3.7: 0 scheming
plot_data = pd.DataFrame({
    'Model': ['GPT o1*'] * 10 + ['Gemini 2.0'] * 10 + ['Claude 3.7'] * 5,
    'Schemed': [1]*8 + [0]*2 + [0]*10 + [0]*5
})

# Calculate mean scheming rate for each model
summary = plot_data.groupby('Model')['Schemed'].mean().reset_index()
summary.columns = ['Model', 'Scheming Rate']

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x='Model', y='Scheming Rate', palette='muted')
plt.title('Observed Scheming Behavior by Model')
plt.ylabel('Proportion of Trials Exhibiting Scheming')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure before displaying
plt.savefig('figure3.png', dpi=150, bbox_inches='tight')
print("Figure saved to figure3.png")

plt.show()

