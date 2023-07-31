
import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pickle import dump


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIRECTORY = "/opt/ml/processing"
TRAIN_DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "mnist_train.csv"
TEST_DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "mnist_test.csv"


def _save_splits(base_directory, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))


def _save_baseline(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need a baseline
    to compute constraints and statistics. This function will save that
    baseline to the disk.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()
        df.to_json(
            baseline_path / f"{split}-baseline.json", orient="records", lines=True
        )


def preprocess(base_directory, train_data_filepath, test_data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train,
    validation, and a test set.
    """
    
    print("started, reading csv...")
    df_train = pd.read_csv(train_data_filepath)
    df_test = pd.read_csv(test_data_filepath)
    
    print("saving baseline...")
    _save_baseline(base_directory, df_train, df_test)
    
    print("sampling dataframe...")
    df_train = df_train.sample(frac=1, random_state=42)
    df_train, df_validation = train_test_split(df_train, test_size=0.3)
    
    print("defining numeric features and transform...")
    numeric_features = df_train.drop("label", axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor)
        ]
    )
    
    print("defining y_ ...")
    y_train = df_train.label
    y_validation = df_validation.label
    y_test = df_test.label
    
    print("defining X_ ...")
    df_train = df_train.drop(["label"], axis=1)
    df_validation = df_validation.drop(["label"], axis=1)
    df_test = df_test.drop(["label"], axis=1)

    X_train = pipeline.fit_transform(df_train)
    X_validation = pipeline.transform(df_validation)
    X_test = pipeline.transform(df_test)
    
    print("concatenating train/valid/test csv...")
    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
    
    print("saving splits...")
    _save_splits(base_directory, train, validation, test)
    print("saving pipeline...")
    _save_pipeline(base_directory, pipeline=pipeline)
    print("finished")
    
if __name__ == "__main__":
    preprocess(BASE_DIRECTORY, TRAIN_DATA_FILEPATH, TEST_DATA_FILEPATH)
