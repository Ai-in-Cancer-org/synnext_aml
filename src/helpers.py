import pandas as pd

def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Infer continuous and binary column names from the data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which the continuous and binary columns are inferred

    Returns
    -------
    Continuous column names and binary column names : tuple[list[str], list[str]]

    """

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    binary_columns = [
        column
        for column in numeric_columns
        if set(df[column].dropna().unique()).issubset({0, 1})
    ]
    continuous_columns = [
        column
        for column in numeric_columns
        if column not in binary_columns
    ]
    return  continuous_columns, binary_columns



def compute_risk_score(
    df: pd.DataFrame,
    cox_coef: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute patient-specific risk scores using coefficients from a fitted
    Cox proportional hazards (CoxPH) model.

    The risk score is calculated as the linear predictor of the Cox model:

        risk_score = Σ(feature_i X coefficient_i)

    where coefficient_i corresponds to the estimated CoxPH coefficient
    for feature_i.
    Only features that are statsically significant were considered for the risk-score
    calculation.

    Parameters
    ----------
    df : pd.DataFrame

        Dataframe containing patient-level features. Column names must
        correspond to feature names used during CoxPH model training.

    cox_coef : pd.DataFrame
        Dataframe containing CoxPH model coefficients. Expected columns are:

        - ``feature`` : feature name
        - ``coef`` : estimated CoxPH coefficient
        - ``p``    : coressponding p-value of the feature

        Additional columns (e.g. p-values, confidence intervals) are ignored.

    Returns
    -------
     pd.DataFrame
        Dataframe containing a single column, ``riskscore``, representing
        risk score for each patient.
    """

    cox_coef=cox_coef[cox_coef["p"]<0.05]

    features = [
        feature
        for feature in cox_coef["feature"].tolist()
        if feature in df.columns
    ]

    if not features:
        raise ValueError("No Cox features found in processed dataframe.")

    coefficients = (
        cox_coef
        .set_index("feature")
        .loc[features, "coef"]
        .astype(float)
    )

    scores = df[features].astype(float).dot(coefficients)

    return pd.DataFrame({"riskscore": scores}, index=df.index)



def riskscore_matching(
    original_score: pd.DataFrame,
    synth_score: pd.DataFrame,
    closest: int = 1,
) -> pd.Index:
    """
    Match Original samples to Synthetic samples based on the nearest risk score.

    For each risk score in `original_score`, the function selects the `closest`
    available rows from `synth_score` whose risk scores have the smallest
    absolute difference. Once selected, synthetic samples are removed from the
    candidate pool and cannot be matched again.

    Parameters
    ----------
    original_score : pandas.DataFrame
        DataFrame containing a ``'riskscore'`` column. Each row has the patient-specific
        risk-score calculated based on the CoxPh regression Co_efficients.

    synth_score : pandas.DataFrame
        DataFrame containing a ``'riskscore'`` column. Each row has the patient-specific
        risk-score calculated based on the CoxPh regression Co_efficients.

    closest : int, optional, default=1
        Number of nearest synthetic samples to select for each original sample.
        Must be a positive integer. Selected samples are removed from the pool
        of available synthetic samples.

    Returns
    -------
    pandas.Index
        Index of the matched rows from `synth_score`.

    """
    if closest < 1:
        raise ValueError("'closest' must be a positive integer.")

    if "riskscore" not in original_score.columns:
        raise KeyError("'original_score' must contain a 'riskscore' column.")

    if "riskscore" not in synth_score.columns:
        raise KeyError("'synth_score' must contain a 'riskscore' column.")

    if len(original_score) * closest > len(synth_score):
        raise ValueError(
            "Not enough rows in 'synth_score' to satisfy the requested matches."
        )

    groupA = original_score.copy()
    groupB = synth_score.copy()

    sampled_indices = []
    B_available = groupB.copy()

    for val in groupA["riskscore"]:
        idx = (
            (B_available["riskscore"] - val)
            .abs()
            .nsmallest(closest)
            .index
        )
        sampled_indices.extend(idx)
        B_available = B_available.drop(idx)

    return groupB.loc[sampled_indices].index
