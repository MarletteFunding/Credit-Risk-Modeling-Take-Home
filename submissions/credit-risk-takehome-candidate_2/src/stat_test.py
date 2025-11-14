import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats.contingency import chi2_contingency

def compare_binary_groups(df, group_col, target_col):

    # Contingency table and chi2 test
    ct = pd.crosstab(df[group_col], df[target_col])
    chi2_stat, chi2_p, _, _ = chi2_contingency(ct)

    # Counts and defaults for each group
    x1 = df.loc[df[group_col] == 1, target_col].sum()
    n1 = (df[group_col] == 1).sum()
    x2 = df.loc[df[group_col] == 0, target_col].sum()
    n2 = (df[group_col] == 0).sum()

    # Default rates
    prop_null = x1 / n1 if n1 > 0 else np.nan
    prop_pop = x2 / n2 if n2 > 0 else np.nan

    # 2-proportion z-test
    z_stat, p_val = proportions_ztest([x1, x2], [n1, n2])

    # Lift
    lift = prop_pop - prop_null

    # Lift CI
    ci_low, ci_upp = confint_proportions_2indep(x2, n2, x1, n1, method='wald')

    # Bundle results
    return {
        "chi2_p": chi2_p,
        "z_test_p": p_val,
        "default_rate_group1": prop_null,
        "default_rate_group0": prop_pop,
        "lift_group0_minus_group1": lift,
        "ci_low": ci_low,
        "ci_upp": ci_upp
    }