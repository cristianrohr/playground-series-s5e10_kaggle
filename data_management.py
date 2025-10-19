from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold



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

    if stratify:
        bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=bins
        )
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return DataSplit(X_train, X_valid, y_train, y_valid)

def get_column_types(x_train: pd.DataFrame):
    num_cols = list(x_train.select_dtypes(include="number"))
    cat_cols = list(x_train.select_dtypes(exclude="number"))
    return cat_cols, num_cols

def kfold_cv(y: pd.Series):
    bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    return list(splitter.split(np.zeros(len(y)), bins))
    

def simple_fe(model):

    num_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean = False))
    ])

    cat_pipeline = Pipeline([
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, selector(dtype_include=np.number)),
        ('cat', cat_pipeline, selector(dtype_exclude=np.number)) 
    ])

    pipeline = Pipeline(
            steps=[
            ('preprocessor', preprocessor),
            ('model', model)
            ]
    )

    return pipeline



def add_features_1_names(transformer, feature_names_in):
    """Top-level helper for FunctionTransformer feature_names_out (pickle-safe)."""
    return list(add_features_1(pd.DataFrame(columns=list(feature_names_in))).columns)

def add_features_2_names(transformer, feature_names_in):
    """Top-level helper for FunctionTransformer feature_names_out (pickle-safe)."""
    return list(add_features_2(pd.DataFrame(columns=list(feature_names_in))).columns)

def add_all_features_names(transformer, feature_names_in):
    """Top-level helper for FunctionTransformer feature_names_out (pickle-safe)."""
    return list(add_all_features(pd.DataFrame(columns=list(feature_names_in))).columns)


def numerical_fe(model):
    # Generate features first; expose dynamic column names to downstream steps
    add_features = FunctionTransformer(
        add_features_1,
        validate=False,
        feature_names_out=add_features_1_names,
    )

    num_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean = False))
    ])

    cat_pipeline = Pipeline([
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, selector(dtype_include=np.number)),
        ('cat', cat_pipeline, selector(dtype_exclude=np.number)) 
    ])

    pipeline = Pipeline(
            steps=[
            ('add_features_1', add_features),
            ('preprocessor', preprocessor),
            ('model', model)
            ]
    )

    return pipeline

def numerical_fe_2(model):
    # Generate features first; expose dynamic column names to downstream steps
    add_features = FunctionTransformer(
        add_features_2,
        validate=False,
        feature_names_out=add_features_2_names,
    )

    num_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean = False))
    ])

    cat_pipeline = Pipeline([
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, selector(dtype_include=np.number)),
        ('cat', cat_pipeline, selector(dtype_exclude=np.number)) 
    ])

    pipeline = Pipeline(
            steps=[
            ('add_features_2', add_features),
            ('preprocessor', preprocessor),
            ('model', model)
            ]
    )

    return pipeline

def numerical_fe_all(model):
    # Generate features first; expose dynamic column names to downstream steps
    add_features = FunctionTransformer(
        add_all_features,
        validate=False,
        feature_names_out=add_all_features_names,
    )

    num_pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean = False))
    ])

    cat_pipeline = Pipeline([
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, selector(dtype_include=np.number)),
        ('cat', cat_pipeline, selector(dtype_exclude=np.number)) 
    ])

    pipeline = Pipeline(
            steps=[
            ('add_all_features', add_features),
            ('preprocessor', preprocessor),
            ('model', model)
            ]
    )

    return pipeline

def add_features_1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'num_lanes' in out and 'speed_limit' in out:
        out['lanes_speed'] = out['num_lanes'] * out['speed_limit']
    if 'curvature' in out and 'speed_limit' in out:
        out['curv_speed'] = out['curvature'] * out['speed_limit']
    if 'num_lanes' in out and 'num_reported_accidents' in out:
        out['acc_per_lane'] = out['num_reported_accidents'] / (out['num_lanes'] + 1)
    if 'num_reported_accidents' in out:
        out['log1p_num_acc'] = np.log1p(out['num_reported_accidents'])
    if 'time_of_day' in out:
        out['is_night'] = out['time_of_day'].isin(['evening','night']).astype(int)
    if 'weather' in out and 'time_of_day' in out:
        out['night_rain'] = ((out['time_of_day'].isin(['evening','night'])) &
                             (out['weather'] == 'rainy')).astype(int)
    return out

def add_features_2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'road_type' in out and 'speed_limit' in out:
        out['road_type_speed'] = out['road_type'].astype(str) + '_' + out['speed_limit'].astype(str)
    if 'weather' in out and 'lighting' in out:
        out['weather_lighting'] = out['weather'].astype(str) + '_' + out['lighting'].astype(str)
    if 'road_type' in out and 'curvature' in out:
        out['road_type_curvature'] = out['road_type'].astype(str) + '_' + out['curvature'].astype(str)
    return out

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_features_1(out)
    out = add_features_2(out)
    return out

def random_search(pl, param_dist, y_train):

    search = RandomizedSearchCV(
        estimator=pl,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings that are sampled
        cv=kfold_cv(y_train),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=42  # for reproducibility
    )

    return search