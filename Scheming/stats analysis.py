import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as models

# Simulated trial-level data based on described frequencies
data = {
    "model": ["GPT_o1"] * 10 + ["Gemini"] * 10 + ["Claude"] * 5,
    "scheming": [1]*8 + [0]*2 + [0]*10 + [0]*5
}

df = pd.DataFrame(data)
df['model'] = pd.Categorical(df['model'], categories=['Gemini', 'GPT_o1', 'Claude'])
model = models.logit("scheming ~ C(model)", data=df).fit()


print(model.summary())
print(f"Pseudo R²: {model.prsquared:.3f}")

odds_ratios = np.exp(model.params)
print("\nOdds Ratios:")
print(odds_ratios)

llf = model.llf     
llnull = model.llnull  # 
lr_stat = 2 * (llf - llnull)
p_value = sm.stats.chisqprob(lr_stat, df=model.df_model)

print(f"\nLikelihood Ratio chi²: {lr_stat:.3f}, p-value: {p_value:.4f}")
