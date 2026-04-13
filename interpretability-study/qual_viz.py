'''
filename: viz.py
date: 12.16.24 (edits)
Authors: Dean Hickman Duilio Lucio
Purpose: Create a plot to visualize qualitative survey data'''

import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Q40_1": [5, 7, 3, 3, 2, 4, 2, 2, 3, 3, 3, 3, 2, 6, 4, 6],
    "Q40_2": [6, 6, 5, 7, 7, 7, 5, 3, 4, 5, 6, 6, 5, 4, 7, 7],
    "Q40_3": [4, 3, 3, 6, 6, 6, 5, 5, 5, 7, 6, 6, 6, 7, 6, 7],
}


means = {key: np.mean(values) for key, values in data.items()}
categories = ["Riddle", "Logic", "General"]
mean_values = list(means.values())


plt.figure(figsize=(8, 6))
plt.bar(categories, mean_values, color=['red', 'green', 'orange'], alpha=0.7)


plt.title('Average confidence for each type', fontsize=16)
plt.ylabel('Average confidence', fontsize=14)
plt.xlabel('Questions', fontsize=14)
plt.ylim(1, 7)

for i, value in enumerate(mean_values):
    plt.text(i, value + 0.1, f'{value:.2f}', ha='center', va='center', fontsize=12)

plt.show()

