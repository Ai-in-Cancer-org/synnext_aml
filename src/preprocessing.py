import pandas as pd
import os
from joblib import load
from pathlib import Path
from hyperimpute.plugins.imputers import Imputers
from helpers import infer_column_types

def preprocessing_riskscoore_matching(
    data: pd.DataFrame,
    preprocessor_path: str | Path = "preprocessor.joblib",
) -> pd.DataFrame:
    """
    Impute missing values and apply a saved preprocessing pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe to preprocess.

    preprocessor_path : str | Path, default="preprocessor.joblib"
        Path to a fitted sklearn-compatible preprocessing pipeline.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with transformed feature names and the original index.
    """

    # impute missing values in the data
    plugin = Imputers().get("missforest", random_state=42)
    x_imputed = plugin.fit_transform(data)


    #preprocess the data using pipeline used in the cox_regression
    preprocessor = load(preprocessor_path)

    x_processed = pd.DataFrame(
        preprocessor.transform(x_imputed),
        columns=preprocessor.get_feature_names_out(),
        index=data.index,
    )

    x_processed.columns = [
        col.split("_")[-1] for col in x_processed.columns
    ]

    return x_processed

def feature_selection( data: pd.DataFrame,
                      drop_columns: list[str]= [],
                      missing_values_ratio: float =50,
                      positive_class_ratio: float=1.0):
    """
    Perform basic feature selection on a DataFrame by removing
    columns that fail to reach missing values percentage ratio and positive class
    frequency.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing continuous and binary features.

    drop_columns : list[str], default=[]

        List of column names to remove before feature selection.

    missing_values_ratio : float, default=50
        Maximum allowed percentage of missing values in a column.
        Columns with a missing-value percentage greater than or equal to
        this threshold are removed.

    positive_class_ratio : float, default=1
        Minimum required percentage of positive values (value == 1) for
        binary columns to be retained. The percentage is calculated using
        non-missing observations only.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing retained columns.

    """
     # Delete the irrelevent columns
    data = data.drop(columns=[col for col in drop_columns if col in data.columns])

    # calculate the missing values ration in each column
    missing_values=pd.DataFrame({"index":data.isna().sum().index,
              "values":data.isna().sum().values,
              "ratio":100*data.isna().sum().values/data.shape[0]})

    #delete columns with missing values more than "missing_values_ratio" value
    missing_columns=missing_values[missing_values.ratio>=missing_values_ratio]["index"].values
    data = data.drop(columns=missing_columns)

    continuous_columns, binary_columns=infer_column_types(data)

    #caculate the frequency
    positive_rate = 100*(data[binary_columns].eq(1).sum() / data[binary_columns].notna().sum())

    #find the columns satisfying the threshold frequency
    positive_class_columns = positive_rate[positive_rate >= positive_class_ratio].index.tolist()

    # final columns
    final_columns=[col for col in data.columns if col in continuous_columns]+positive_class_columns
    return data[final_columns]


if __name__=="__main__":
    filepath=os.path.dirname(__file__)
    scalarpath=os.path.join(filepath, "..", "models", "preprocessing","preprocessor.joblib")
    datapath=os.path.join(filepath, "..", "data", "raw","data.csv")
    data=pd.read_csv(datapath)
    preprocessing_riskscoore_matching(data, scalarpath)
