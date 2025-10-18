from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)  # if you use numpy RNG elsewhere

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    test: Optional[pd.DataFrame] = None


def load_data(train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load train (and optional test) CSVs into DataFrames."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df


def make_split(
    df: pd.DataFrame,
    target: str,
    drop: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DataSplit:
    """Split a labeled dataframe into train/validation sets.

    Parameters
    - df: DataFrame containing features and the target column
    - target: target column name
    - drop: columns to drop
    - test_size: fraction for validation split
    - random_state: random seed for reproducibility
    - stratify: whether to stratify by target
    """
    y = df[target]
    X = df.drop(columns=[target, drop])
    strat = y if stratify else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return DataSplit(X_train, X_valid, y_train, y_valid)

def get_column_types(x_train: pd.DataFrame):
    num_cols = list(x_train.select_dtypes(include="number"))
    cat_cols = list(x_train.select_dtypes(exclude="number"))
    return cat_cols, num_cols


def simple_fe(categorical_features: list, numerical_features: list):

    num_pipeline = Pipeline([
        ('num', 'passthrough')
    ])

    cat_pipeline = Pipeline([
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features) 
    ])

    pipeline = Pipeline(
            steps=[
            ('preprocessor', preprocessor),
            ('rf', RandomForestRegressor(random_state=42, 
                n_jobs=1))
            ]
    )

    return pipeline

