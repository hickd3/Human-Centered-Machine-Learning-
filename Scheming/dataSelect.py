'''
filename: dataSelect.py
Authors: Dean Hickman
Purpose: Randomly sample 36 patient profiles from insurance.csv for LLM evaluation trials.
'''

import pandas as pd

# Load the dataset 
df = pd.read_csv('insurance.csv')

# Randomly select 36 rows 
random_rows = df.sample(n=36, random_state=42)

# Save selected profiles for use in LLM evaluation prompts
random_rows.to_csv('selected_rows.csv', index=False)

print(f"Sampled {len(random_rows)} rows. Saved to selected_rows.csv")
print(random_rows)

