import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame for plotting
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
plt.show()

