# Playground Series S5E10 — Accident Risk (Kaggle)

Lightweight pipeline to load data, split train/validation, build an sklearn preprocessing + model pipeline, and generate predictions for Kaggle Playground Series S5E10.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── fe_simple/               # optional cached splits
│       ├── x_train.csv
│       ├── x_valid.csv
│       ├── y_train.csv
│       └── y_valid.csv
├── models/
│   └── model_rf_fe_simple.pkl   # saved model (example)
├── data_management.py           # data loading + splitting utilities
├── train.py                     # evaluation helper(s)
├── main.py                      # script to split/train/predict
└── requirements.txt
```

## Setup

- Python 3.10+ (tested with 3.12)
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place Kaggle files in `data/` with the exact names:
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

The competition provides an `id` column and a target (here referred to as `accident_risk`). The `id` column should not be used as a feature.

## Data Utilities

`data_management.py` provides:

```py
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    test: Optional[pd.DataFrame] = None

def load_data(train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]

def make_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DataSplit
```

Usage:

```py
from data_management import load_data, make_split

train_df, test_df = load_data("data/train.csv", "data/test.csv")
# drop non-feature columns like id before splitting
train_df = train_df.drop(columns=["id"])  # keep target

ds = make_split(train_df, target="accident_risk")
X_train, X_valid = ds.X_train, ds.X_valid
y_train, y_valid = ds.y_train, ds.y_valid
```

## Preprocessing + Model (sklearn)

To avoid data leakage, always split first, then fit preprocessing only on the training data. Put preprocessing inside a `Pipeline` (and `ColumnTransformer`) so CV and refits are safe.

Example baseline:

```py
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, selector(dtype_include=np.number)),
    ("cat", cat_pipe, selector(dtype_exclude=np.number)),
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000)),  # choose your model
])

pipeline.fit(X_train, y_train)
val_pred = pipeline.predict_proba(X_valid)[:, 1]  # for classification metrics
```

## Evaluation

`train.py` currently exposes:

```py
from sklearn.metrics import root_mean_squared_error

def evaluate_model(yreal, ypred):
    return root_mean_squared_error(yreal, ypred)
```

Notes:
- RMSE is a regression metric. If the target is binary, prefer metrics such as AUC (`roc_auc_score`) or log loss (`log_loss`) with probability predictions (`predict_proba`), or accuracy/F1 with class predictions (`predict`).
- Ensure shapes match: `y_true` and `y_pred` must be 1D arrays of the same length. For probability-based metrics, pass a 1D probability vector (e.g., `pred[:, 1]`).

## Running

The provided `main.py` is a simple driver that can:
- Split and cache train/valid to CSVs under `data/fe_simple/`
- Train and save a model under `models/`
- Load a saved model and evaluate on the validation split

Edit `run_type` in `main.py` to switch modes:

- `"train"`: fits a model and saves to `models/`. You should implement or replace the placeholder feature engineering helpers (see next section) with the pipeline shown above.
- `"predict"`: loads a saved model and evaluates on the validation split.

Example flow:

1) Prepare split caches (optional)
```py
# in main.py
run_type = "train"
# ensure you drop id and build a valid sklearn pipeline
```

2) Train a model
```bash
python main.py
```

3) Predict on validation
```py
# in main.py
run_type = "predict"
```
```bash
python main.py
```

4) Predict on test for Kaggle

Use the fitted pipeline to call `predict_proba(test_df)[:, 1]` and write to a submission file with the required columns (`id`, `accident_risk`). Example:

```py
sub = pd.read_csv("data/sample_submission.csv")
proba = pipeline.predict_proba(test_df.drop(columns=["id"]))[:, 1]
sub["accident_risk"] = proba
sub.to_csv("submission.csv", index=False)
```

## Important Gotchas

- Split first, then fit preprocessing on training only (avoid leakage).
- Keep `X` as a pandas DataFrame so `ColumnTransformer` can select columns by name.
- When specifying columns to `ColumnTransformer`, pass plain Python lists or use `make_column_selector`.
- For classification, do not evaluate RMSE on class probabilities of shape `(n_samples, 2)`—use `pred[:, 1]` with a classification metric.
- If you see `TypeError: cannot unpack non-iterable DataSplit`, access attributes: `ds.X_train`, `ds.y_train`, etc.

## Roadmap / Next Steps

- Replace placeholder feature engineering in `main.py` with the Pipeline + ColumnTransformer pattern above (or implement `simple_fe`/`get_column_types`).
- Add hyperparameter tuning via `GridSearchCV`/`RandomizedSearchCV`.
- Implement proper metric(s) matching the challenge target (AUC/log loss for binary classification).
- Add k-fold cross-validation and model ensembling if desired.

## License

No license specified. Use at your own discretion.

