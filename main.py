from data_management import load_data, make_split
from data_management import simple_fe, get_column_types, numerical_fe, focused_fe
from train import evaluate_model
import pickle
import pandas as pd
import os
from data_management import random_search, numerical_fe_all, numerical_fe_2
import joblib
import lightgbm as lgb
import xgboost as xgb
baseline_rmse = 0.05937

run_type = "predict" # train or predict

sub_name = "xgb_focused"
#model_name = "rf"
model_name = "xgb"
#fe_folder = "fe_simple"
#fe_folder = "fe_num_new_features"
fe_folder = "fe_focused"
if fe_folder not in os.listdir("data"):
    os.mkdir(f"data/{fe_folder}")
model_folder = f"models"
split_data = True

if run_type == "predict":
    split_data = False

if split_data:
    train, x_test = load_data("data/train.csv", "data/test.csv")
    ds = make_split(train, "accident_risk", "id")
    x_train, x_valid, y_train, y_valid = ds.X_train, ds.X_valid, ds.y_train, ds.y_valid

    x_train.to_csv(f"data/{fe_folder}/x_train.csv", index = False)
    x_valid.to_csv(f"data/{fe_folder}/x_valid.csv", index = False)
    y_train.to_csv(f"data/{fe_folder}/y_train.csv", index = False)
    y_valid.to_csv(f"data/{fe_folder}/y_valid.csv", index = False)
else:
    x_train = pd.read_csv(f"data/{fe_folder}/x_train.csv")
    x_valid = pd.read_csv(f"data/{fe_folder}/x_valid.csv")
    y_train = pd.read_csv(f"data/{fe_folder}/y_train.csv").iloc[:,0]
    y_valid = pd.read_csv(f"data/{fe_folder}/y_valid.csv").iloc[:,0]

if run_type == "train":
    cat_cols, num_cols = get_column_types(x_train)

    #pipeline = simple_fe(cat_cols, num_cols)
    #pipeline = numerical_fe(x_train)

    """
    model = lgb.LGBMRegressor(random_state=42, n_jobs=1)
    param_dist = {
        "model__n_estimators": [500, 1000, 2000, 3000],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "model__max_depth": [5, 7, 9, 11],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    }
    """
    model = xgb.XGBRegressor(random_state=42, n_jobs=1)
    
    param_dist = {  
        "model__n_estimators": [500, 800, 1000, 1500],
        "model__learning_rate": [0.005, 0.01, 0.05],
        "model__max_depth": [5, 7, 9],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_lambda": [3.0, 5.0, 7.0],
    }

    pl = focused_fe(model)
    pipeline = random_search(pl, param_dist, y_train)

    model_fitted = pipeline.fit(x_train, y_train)

    print("CV best (neg RMSE):", model_fitted.best_score_)
    print("Best params:", model_fitted.best_params_)
    best_model = model_fitted.best_estimator_

    joblib.dump(model_fitted, f"{model_folder}/model_{model_name}_{fe_folder}.joblib")    

if run_type == "predict":
    model_to_use = joblib.load(open(f"{model_folder}/model_{model_name}_{fe_folder}.joblib", "rb"))
    best_pipe = model_to_use.best_estimator_
    print("Best params:", model_to_use.best_params_)
    
    val_preds = best_pipe.predict(x_valid)
    rmse_res = evaluate_model(val_preds, y_valid)

    print("Validation RMSE:", rmse_res)

    print(round(rmse_res,4))
    print(baseline_rmse)
    if round(rmse_res,4) <= round(baseline_rmse,4):
        print("Vamos a predecir test")
        test_data = pd.read_csv("data/test.csv")
        
        X_full = pd.concat([x_train, x_valid], axis=0)
        y_full = pd.concat([y_train, y_valid], axis=0)
        best_pipe.fit(X_full, y_full)

        test_preds = best_pipe.predict(test_data.drop(columns=["id"]))

        submission = pd.DataFrame({"id": test_data["id"], "accident_risk": test_preds})
        submission.to_csv(f"submissions/{sub_name}.csv", index=False)





