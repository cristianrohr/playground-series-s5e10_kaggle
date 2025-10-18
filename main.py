from data_management import load_data, make_split
from data_management import simple_fe, get_column_types, numerical_fe
from train import evaluate_model
import pickle
import pandas as pd

baseline_rmse = 0.05937

run_type = "predict" # train or predict

sub_name = "xgb"
#model_name = "rf"
model_name = "xgb"
#fe_folder = "fe_simple"
fe_folder = "fe_num"
model_folder = f"models/"
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
    pipeline = numerical_fe(cat_cols, num_cols)

    model_fitted = pipeline.fit(x_train, y_train)
    with open(f"{model_folder}/model_{model_name}_{fe_folder}.pkl", "wb") as f:
        pickle.dump(model_fitted, f)

if run_type == "predict":
    model_to_use = pickle.load(open(f"{model_folder}/model_{model_name}_{fe_folder}.pkl", "rb"))
    val_preds = model_to_use.predict(x_valid)
    rmse_res = evaluate_model(val_preds, y_valid)

    print(round(rmse_res,4))
    print(baseline_rmse)
    if round(rmse_res,4) <= round(baseline_rmse,4):
        print("Vamos a predecir test")
        test_data = pd.read_csv("data/test.csv")
        test_preds = model_to_use.predict(test_data.drop(columns=["id"]))

        submission = pd.DataFrame({"id": test_data["id"], "accident_risk": test_preds})
        submission.to_csv(f"submissions/{sub_name}.csv", index=False)




