'''
filename: stats_analysis.py
Authors: Dean Hickman, Duilio Lucio
Purpose: Binary logistic regression testing whether scheming behavior is associated
         with model identity, using simulated trial-level data derived from observed
         evaluation frequencies and Meinke et al. (2025) reported rates.
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as models
from scipy.stats import chi2

# ── Simulated trial-level data ────────────────────────────────────────────────
# Frequencies: GPT o1* = 8/10 scheming, Gemini = 0/10, Claude = 0/5
# (derived from evaluation observations and Meinke et al. 2025 reported rates)
data = {
    "model": ["GPT_o1"] * 10 + ["Gemini"] * 10 + ["Claude"] * 5,
    "scheming": [1]*8 + [0]*2 + [0]*10 + [0]*5
}

df = pd.DataFrame(data)
df['model'] = pd.Categorical(df['model'], categories=['Gemini', 'GPT_o1', 'Claude'])

# NOTE: With perfect class separation in Gemini and Claude (all zeros), MLE
# will not converge to finite coefficients and statsmodels will emit a
# ConvergenceWarning. This is expected and documented. The LLR statistic
# and pseudo-R² remain valid summary measures; individual coefficients
# and odds ratios should NOT be interpreted due to separation.
model = models.logit("scheming ~ C(model)", data=df).fit()

print(model.summary())
print(f"Pseudo R²: {model.prsquared:.3f}")

odds_ratios = np.exp(model.params)
print("\nOdds Ratios (unreliable due to quasi-separation — see note above):")
print(odds_ratios)

# ── Likelihood Ratio test ─────────────────────────────────────────────────────
# CORRECTION: sm.stats.chisqprob() was removed in statsmodels >=0.14.
# Use scipy.stats.chi2.sf() instead (survival function = 1 - CDF).
llf    = model.llf
llnull = model.llnull
lr_stat = 2 * (llf - llnull)
p_value = chi2.sf(lr_stat, df=model.df_model)   

print(f"\nLikelihood Ratio chi²: {lr_stat:.3f}, p-value: {p_value:.4f}")
