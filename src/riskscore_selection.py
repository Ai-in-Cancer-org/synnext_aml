from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from hyperimpute.plugins.imputers import Imputers
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd


def range_based_filter(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    range_columns: list[str]=['AGE', 'WBC', 'HB', 'PLT', 'LDH', 'BMB', 'PBB'],
    survival_time_column: str ="EFSTM",
) -> pd.DataFrame:

    """
    Filter synthetic patients based on the observed ranges of the original cohort.

    Synthetic observations are retained only if their values fall within
    the minimum and maximum values observed in the original dataset and also
    rows with negative survival times are ignored

    Parameters
    ----------
    original_df : pd.DataFrame
        Original patient cohort used to define acceptable feature ranges.

    synthetic_df : pd.DataFrame
        Synthetic patient cohort to be filtered.

    range_columns : list[str], default=['AGE', 'WBC', 'HB', 'PLT', 'LDH', 'BMB', 'PBB']
        Clinical variables whose values must lie within the observed
        range of the original cohort.

    survival_time_column : str, default="EFSTM"
        Survival endpoint used for validation of survival time.

    Returns
    -------
    pd.DataFrame
        Filtered synthetic cohort containing only patients satisfying
        all range constraints.
    """

    selected = synthetic_df.copy()

    # Filter Synthetic data based ranges from continuous columns
    for column in range_columns:
        if column not in original_df.columns or column not in selected.columns:
            continue

        lower = original_df[column].min()
        upper = original_df[column].max()

        selected = selected[
            selected[column].between(lower, upper) | selected[column].isna()
        ].copy()

    # Only take rows with positive survival time
    selected = selected[selected[survival_time_column] > 0].copy()

    return selected


def selection_riskscore_based(original_df, synthetic_df, survival_time_column="EFSTM", cox_coef="full", closest=1, soraml=False):
    # load cox_coefficients
    cox_coef=pd.read_csv("../results/EFS_COX_RS.csv")

    continuous_columns, binary_columns=infer_column_types(synthetic_df)
    continuous_columns.pop(survival_time_column)
    #range filtering
    syn_preselected=range_based_filter(original_df,synthetic_df, continuous_columns)

    #select significant features
    features=[col for col in synthetic_df.columns]# if col in df.columns]
    original_df_processed=preprocessing_pre(original_df[[col for col in features if col in original_df.columns]])
    synthetic_df_processed=preprocessing_pre(syn_preselected[[col for col in features if col in syn_preselected.columns]])

    # risk score
    synth_score=compute_risk_score(synthetic_df_processed, cox_coef, features)
    original_score=compute_risk_score(original_df_processed, cox_coef, features)
    selection_indicies=riskscore_matching(original_score, synth_score, closest=closest)
    syn_selected=syn_preselected.loc[selection_indicies,:]

    p_value=log_rank_analysis(syn_selected, original_df, survival=survival)
    print(f"p value of log-rank test {p_value}")
    # kp_curve=KM_curve(original_df,syn_selected, survival=survival)
    # return f"Selected {syn_selected.shape[0]} patients from {syn_preselected.shape[0]}"
    syn_selected["AGE"]=np.floor(syn_selected["AGE"]).astype("int")
    return syn_selected, p_value
