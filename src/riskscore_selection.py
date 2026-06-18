
from hyperimpute.plugins.imputers import Imputers
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd
from helpers import *
from joblib import dump, load
from pathlib import Path
from typing import Tuple

from survival_metrics import log_rank_analysis
from helpers import riskscore_matching
from preprocessing import preprocessing_riskscoore_matching

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


def selection_riskscore_based(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    closest: int = 1,
    survival_time_column: str = "EFSTM",
    survival_event_column : str = "EFSSTAT",
    cox_coef_path: str | Path = "../results/EFS_COX_RS.csv",
    preprocessor_path: str | Path = "preprocessor.joblib",
) -> Tuple[pd.DataFrame, float]:
    """
    Select synthetic samples that best match the original cohort based on Cox risk scores.

    The function:
    1. Loads Cox model coefficients.
    2. Infers continuous and binary columns from the synthetic data.
    3. Applies range-based filtering to remove synthetic samples outside the
       original data distribution.
    4. Applies preprocessing and imputation.
    5. Computes Cox-based risk scores for both original and synthetic samples.
    6. Selects synthetic samples whose risk scores most closely match the original cohort.
    7. Performs a log-rank test between selected synthetic and original survival curves.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original real cohort data.

    synthetic_df : pd.DataFrame
        Synthetic cohort data from which samples will be selected.

    closest : int, default=1
        Number of closest synthetic samples to select per original sample.

    survival_time_column : str, default="EFSTM"
        Name of the survival time column to exclude from continuous feature filtering.
    
    survival_event_column : str, default="EFSSTAT"
        Survival outcome event column name

    cox_coef_path : str | Path, default="../results/EFS_COX_RS.csv"
        Path to the CSV file containing Cox model coefficients.

    preprocessor_path : str | Path, default="preprocessor.joblib"
        Path to the fitted preprocessing pipeline.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        synthetic_cohort_selected : pd.DataFrame
            Selected synthetic samples matched by risk score.

        logrank_p_value : float
            Log-rank test p-value comparing selected synthetic and original survival data.
    """
    
    #load pre trained CoxPh model co-efficients
    cox_coef = pd.read_csv(cox_coef_path)

    #Find the continuous column
    continuous_columns, binary_columns = infer_column_types(synthetic_df)

    #Delete Outcome Variable
    if survival_time_column in continuous_columns:
        continuous_columns.remove(survival_time_column)

    # Range based Filtering
    synthetic_preselected = range_based_filter(
        original_df,
        synthetic_df,
        continuous_columns,survival_time_column
    )
    
    # preprocees the real data for riskscore calculation
    original_df_processed = preprocessing_riskscoore_matching(
        original_df,
        preprocessor_path=preprocessor_path,
    )

    # preprocees the synthetic data for riskscore calculation
    synthetic_df_processed = preprocessing_riskscoore_matching(
        synthetic_preselected,
        preprocessor_path=preprocessor_path,
    )

    # compute patient specific risk scores for real and synthetic data
    synthetic_riskscore = compute_risk_score(synthetic_df_processed, cox_coef)
    original_riskscore = compute_risk_score(original_df_processed, cox_coef)

    # Match real and synthetic data based on risk score comparison
    selection_indices = riskscore_matching(
        original_riskscore,
        synthetic_riskscore,
        closest=closest,
    )
    
    synthetic_cohort_selected = synthetic_preselected.loc[selection_indices, :].copy()

    # calculate the log-rank p-vale between selected synthetic cohorts and real data
    p_value = log_rank_analysis(
        synthetic_cohort_selected,
        original_df,
        survival_time_column=survival_time_column,
        survival_event_column=survival_event_column,
    )

    #convert age column into integer
    if "AGE" in synthetic_cohort_selected.columns:
        synthetic_cohort_selected["AGE"] = np.floor(synthetic_cohort_selected["AGE"]).astype(int)
    synthetic_cohort_selected[binary_columns]=synthetic_cohort_selected[binary_columns].astype("int")
    return synthetic_cohort_selected, p_value
