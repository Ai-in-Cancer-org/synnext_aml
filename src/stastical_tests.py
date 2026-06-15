from scipy.stats import ttest_ind # Welch’s t-test
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency, fisher_exact 
import pandas as pd
import numpy as np


def chi2_test(df1: pd.DataFrame,
    df2: pd.DataFrame,
    list_of_columns: list[str]):

    n1 = len(df1)
    n2 = len(df2)
    
        #Value counts per group
    counts1 = df1[list_of_columns].value_counts()
    counts2 = df2[list_of_columns].value_counts()
    
    # 2) All categories present in either dataframe
    categories = sorted(set(counts1.index) | set(counts2.index))
    
    results = {}

    
    for cat in categories:
        # Counts for this category in each group
        cat1 = counts1.get(cat, 0)
        cat2 = counts2.get(cat, 0)
    
        # "Not this category" counts
        not_cat1 = n1 - cat1
        not_cat2 = n2 - cat2
    
        # 2x2 table:
        # [ [cat in group1, cat in group2],
        #   [not-cat in group1, not-cat in group2] ]
        table = [[cat1, cat2],
                 [not_cat1, not_cat2]]
        chi2, p_value, dof, expected = chi2_contingency(table)
    
        results[cat]=p_value
    return results