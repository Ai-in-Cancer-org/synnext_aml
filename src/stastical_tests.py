from scipy.stats import ttest_ind # Welch’s t-test
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency, fisher_exact 
import pandas as pd
import numpy as np


def fisher_exact_binary(df1: pd.DataFrame,
    df2: pd.DataFrame,
    list_of_columns: list[str]):

     # 1️⃣ Count positive class (1s)
    counts_real = df1[list_of_columns].sum()
    counts_synth = df2[list_of_columns].sum()
    
    # 2️⃣ Compute p-values per feature (Fisher exact test)
    pvals = []
    for col in list_of_columns:
        pos_real = counts_real[col]
        pos_synth = counts_synth[col]
        neg_real = len(df1) - pos_real
        neg_synth = len(df2) - pos_synth
        table = np.array([[pos_real, neg_real],
                          [pos_synth, neg_synth]])
        # print(col,table)
        _, p = fisher_exact(table, alternative="two-sided")
        pvals.append(p)
    
    # 3️⃣ Combine counts + p-values
    count_df = pd.DataFrame({
        "% Positive Real Data": np.round(100*counts_real/df1.shape[0],3),
        "% Positive Synthetic Data": np.round(100*counts_synth/df2.shape[0], 3),
        "p_value": np.round(pvals, 3)
    }, index=list_of_columns)
    count_df=count_df.sort_index(ascending=False)
    
    return count_df