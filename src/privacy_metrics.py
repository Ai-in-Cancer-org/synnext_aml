import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from helpers import infer_column_types
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score

def exact_match_count(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    n_bins: int = 5
    ) -> int:

    """
    Compute the number of exact feature-pattern matches between two datasets.

    Continuous variables are discretized into quantile bins using the
    reference dataset. The same bin edges are then applied to the comparison
    dataset. Exact matches are evaluated on the combination of binary features
    and discretized continuous features.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Dataset used to define quantile bin edges.

    comparison_df : pd.DataFrame
        Dataset whose rows are checked for matching patterns.

    n_bins : int
        Number of quantile bins, default=5

    Returns
    -------
    int:
            Number of rows in comparison_df whose pattern exists in reference_df.
    """

    ref = reference_df.copy()
    comp = comparison_df.copy()

    bin_cols = []

    continuous_columns, binary_columns= infer_column_types(ref)

    for col in continuous_columns:
        bin_col = f"{col}_bin"
        bin_cols.append(bin_col)

        ref[bin_col], bin_edges = pd.qcut(
            ref[col],
            q=n_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )

        comp[bin_col] = pd.cut(
            comp[col],
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )

    match_cols = list(binary_columns) + bin_cols

    ref_patterns = set(
        ref[match_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    comp_patterns = (
        comp[match_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    matches = comp_patterns.isin(ref_patterns)

    match_count = int(matches.sum())

    return match_count



def dcr_euclidean(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame
    ) -> float:

    """
    Compute Distance to Closest Record (DCR) using Euclidean distance.

    For each row in comparison_df, the function finds the closest row in
    reference_df and reports the median distance.

    Parameters
    ----------
    real_df : pd.DataFrame
        Dataset used as the reference population.

    synthetic_df : pd.DataFrame
        Dataset whose rows are compared against the reference dataset.

    Returns
    -------
    float
        median_dcr :
            Median distance to the closest reference record.
    """
    # Identify feature columns for distance computation
    continuous_columns, binary_columns = infer_column_types(real_df)
    feature_cols = list(continuous_columns) + list(binary_columns)

    real_scaled = real_df[feature_cols].copy()
    synthetic_scaled = synthetic_df[feature_cols].copy()

    # Scale continuous variables while preserving binary features
    scaler = RobustScaler()
    real_scaled[continuous_columns] = scaler.fit_transform(
        real_df[continuous_columns]
    )
    synthetic_scaled[continuous_columns] = scaler.transform(
        synthetic_df[continuous_columns]
    )

    # Compute pairwise Euclidean distances between synthetic and real records
    dist_matrix = pairwise_distances(
        synthetic_scaled[feature_cols],
        real_scaled[feature_cols],
        metric="euclidean",
    )

    # Return median DCR
    closest_distances= dist_matrix.min(axis=1)
    median_dcr = float(np.median(closest_distances))

    return median_dcr

def nndr_euclidean(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> float:
    """
    Compute the median Nearest Neighbour Distance Ratio (NNDR).

    For each synthetic record, the distance to the nearest and second-nearest
    records in the real dataset is computed using Euclidean distance. The
    nearest neighbour distance ratio is then calculated as:

        NNDR = d1 / d2

    where d1 is the distance to the closest real record and d2 is the
    distance to the second-closest real record. The function returns the
    median NNDR across all synthetic records.

    Parameters
    ----------
    real_df : pd.DataFrame
        Dataset used as the reference population.

    synthetic_df : pd.DataFrame
        Dataset whose rows are compared against the reference dataset.

    Returns
    -------
    float
        median_nndr :
            Median nearest neighbour distance ratio across all synthetic
            records.
    """

    # Identify feature columns for distance computation
    continuous_columns, binary_columns = infer_column_types(real_df)
    feature_cols = list(continuous_columns) + list(binary_columns)

    real_scaled = real_df[feature_cols].copy()
    synthetic_scaled = synthetic_df[feature_cols].copy()

    # Scale continuous variables while preserving binary features
    scaler = RobustScaler()
    real_scaled[continuous_columns] = scaler.fit_transform(
        real_df[continuous_columns]
    )
    synthetic_scaled[continuous_columns] = scaler.transform(
        synthetic_df[continuous_columns]
    )

    # Compute pairwise Euclidean distances between synthetic and real records
    dist_matrix = pairwise_distances(
        synthetic_scaled[feature_cols],
        real_scaled[feature_cols],
        metric="euclidean",
    )

    # Extract nearest and second-nearest neighbour distances
    sorted_distances = np.sort(dist_matrix, axis=1)
    nearest_distances = sorted_distances[:, 0]
    second_nearest_distances = sorted_distances[:, 1]

    # Compute nearest neighbour distance ratio
    nndr = nearest_distances / second_nearest_distances

    # Return median NNDR
    median_nndr = float(np.median(nndr))

    return median_nndr


def membership_inference_attack(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute membership inference attack performance using nearest-neighbour distance.

    For each record in the real and test datasets, the function computes the
    Euclidean distance to its closest synthetic record. Records closer to the
    synthetic dataset are assigned higher membership likelihood scores.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real training dataset representing member records.

    synthetic_df : pd.DataFrame
        Synthetic dataset used as the reference dataset.

    test_df : pd.DataFrame
        Holdout real dataset representing non-member records.

    Returns
    -------
    pd.DataFrame
        DataFrame containing membership inference ROC-AUC and accuracy.
    """

    # Identify feature columns for distance computation
    continuous_columns, binary_columns = infer_column_types(real_df)
    all_cols = list(binary_columns) + list(continuous_columns)

    # Scale continuous variables while preserving binary features
    scaler = RobustScaler()

    real_scaled = real_df.copy()
    synthetic_scaled = synthetic_df.copy()
    test_scaled = test_df.copy()

    real_scaled[continuous_columns] = scaler.fit_transform(
        real_df[continuous_columns]
    )

    synthetic_scaled[continuous_columns] = scaler.transform(
        synthetic_df[continuous_columns]
    )

    test_scaled[continuous_columns] = scaler.transform(
        test_df[continuous_columns]
    )

    # Compute pairwise Euclidean distances to synthetic records
    train_dist = pairwise_distances(
        real_scaled[all_cols],
        synthetic_scaled[all_cols],
        metric="euclidean",
    )

    test_dist = pairwise_distances(
        test_scaled[all_cols],
        synthetic_scaled[all_cols],
        metric="euclidean",
    )

    # Extract nearest synthetic neighbour distance
    train_scores = train_dist.min(axis=1)
    test_scores = test_dist.min(axis=1)

    # Create membership labels: 1 = member, 0 = non-member
    y_true = np.concatenate(
        [
            np.ones(len(train_scores)),
            np.zeros(len(test_scores)),
        ]
    )

    # Use negative distance because lower distance implies higher membership likelihood
    attack_scores = np.concatenate(
        [
            -train_scores,
            -test_scores,
        ]
    )

    # Compute membership inference ROC-AUC
    roc_auc = float(roc_auc_score(y_true, attack_scores))

    # Compute threshold-based attack accuracy
    threshold = np.median(attack_scores)
    y_pred = (attack_scores >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))

    return pd.DataFrame(
        {
            "roc_auc": [roc_auc],
            "accuracy": [accuracy],
        }
    )
