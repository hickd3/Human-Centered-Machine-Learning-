import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/deanehickman/Downloads/insurance.csv')

# Randomly select 36 rows
random_rows = df.sample(n=36, random_state=42)  # random_state for reproducibility

# Optionally, you can save the selected rows to a new CSV file
random_rows.to_csv('random_selected_rows.csv', index=False)

# Show the selected rows
print(random_rows)

