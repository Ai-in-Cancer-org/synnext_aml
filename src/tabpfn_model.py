from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


def generate_synthetic_data_experiments(
    real_data: pd.DataFrame,
    n_synthetic_patients: int=10000,
    test_size: float = 0.25,
    split_random_state: int=42,
    drop_columns: list[str]= []
) -> tuple[pd.DataFrame]:
    """
    Generate synthetic patient data using TabPFN across multiple experiments and seeds
    and stores in synthetic data in specified output directory.

    Parameters
    ----------
    real_data : pd.DataFrame
        Original patient-level dataframe.

    n_synthetic_patients : int
        Number of synthetic patients to generate per experiment.

    test_size : float, default=0.25
        Fraction of the data used as test set.

    split_random_state: int, default=42,
        random state for the train test split of the data

    drop_columns : list[str]
        Columns to remove before fitting the synthetic data generator.

    Returns
    -------
    tuple[pd.DataFrame]
        Returns three dataframes:
        1) Synthetic data generated
        2) Training data used in model generation
        3) Held out data for further evaluation
    """

    # load the data
    X=real_data.copy()

    if not set(drop_columns).issubset(X.columns):
        missing = set(drop_columns) - set(X.columns)
        raise ValueError(f"Dataframe does not contain the following drop columns: {missing}")
    else:
        X.drop(columns=drop_columns, inplace=True)


    #data prepare
    X_train, X_test=train_test_split(X, random_state=split_random_state, test_size=test_size)

    # Create the unsupervised model
    clf = TabPFNClassifier()
    reg = TabPFNRegressor()
    model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

    # Fit the model on data without labels
    model.fit(X_train)

    #generate the Synthetic data
    X_synthetic = model.generate_synthetic_data(n_synthetic_patients)
    X_synthetic_np = X_synthetic.numpy() #convert to Numpy array
    synthetic_data = pd.DataFrame(X_synthetic_np, columns=X_train.columns) #convert to a dataframe

    return synthetic_data, X_train, X_test

if __name__=="__main__":

    # create a folder to save the data using runtime
    file_path=os.path.dirname(__file__)
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name=os.path.join(file_path, "..", "data", "synthetic",folder_name)
    os.makedirs(output_dir_name, exist_ok=True)

    #load the data
    data_path=os.path.join(file_path, "..", "data", "raw", "data.csv")
    data=pd.read_csv(data_path)

    #generate synthetic data
    synthetic_data, X_train, X_test= generate_synthetic_data_experiments(data)

    # save the data
    synthetic_data.to_csv(f"{output_dir_name}/synthetic_data.csv", index=False)
    X_train.to_csv(f"{output_dir_name}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir_name}/X_test.csv", index=False)
