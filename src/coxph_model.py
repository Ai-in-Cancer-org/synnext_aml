from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

from hyperimpute.plugins.imputers import Imputers
from helpers import infer_column_types

def preprocessing_coxph(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    """
    Preprocess feature data for Cox Proportional Hazards modeling.

    Missing values are imputed using the MissForest algorithm. Continuous
    variables are identified automatically and scaled using a RobustScaler,
    while non-continuous variables are passed through unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataframe excluding survival outcome columns.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with imputed values and scaled continuous
        features, ready for Cox Proportional Hazards model fitting.
    Pipeline
        Preprocessor pipeline used for the COX propotional hazard modeling
        
    """

    # Impute missing values using MissForest
    plugin = Imputers().get("missforest", random_state=42)
    X_impute = plugin.fit_transform(df)

    # Identify continuous columns for scaling
    continuous_columns, binary_columns = infer_column_types(X_impute)

    # Scale continuous variables and retain remaining columns
    preprocessor = make_column_transformer(
        (RobustScaler(), continuous_columns)
    )

    # Transform data and return as a DataFrame
    X_continuous = pd.DataFrame(
        preprocessor.fit_transform(X_impute),
        columns=preprocessor.get_feature_names_out(),
    )

    # rename to original column names
    X_continuous.columns=[col.split("_")[-1] for col in X_continuous.columns]

    #merge binary columns
    X_processed=pd.concat([X_continuous, X_impute[binary_columns]], axis=1)

    return X_processed, preprocessor


def coxph_model( df: pd.DataFrame,
                survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT",
    )-> tuple[pd.DataFrame, Pipeline]:
    """
    Compute optimism between real and synthetic survival data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with survival time and survival event values

    survival_time_column : str, default="EFSTM"
        Survival outcome time column name.

    survival_event_column : str, default="EFSSTAT"
        Survival outcome event column name

    Returns
    -------
    pd.DataFrame
        DataFrame containing cox-correlation co-efficients for each feature
        along with thier stastical significance values.
    Pipeline
        Preprocessor pipeline used for the COX propotional hazard modeling
    """

    #preprocessing of the data
    X=df.drop(columns=[survival_event_column,survival_time_column])
    y=df[[survival_event_column,survival_time_column]]
    X_processed, preprocessor=preprocessing_coxph(X)
    X_COX=pd.concat([X_processed,y], axis=1)

    # Fit Cox model
    coxph = CoxPHFitter()
    coxph.fit(X_COX, duration_col=survival_time_column, event_col=survival_event_column)
    coxph_df= coxph.summary.copy()

    return coxph_df, preprocessor
