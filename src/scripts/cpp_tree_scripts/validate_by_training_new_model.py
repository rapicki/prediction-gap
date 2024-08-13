from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
from src.decision_tree.shap_wrapper import ShapWrapper
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper


def train_and_save_model(param: dict, steps: int, dtrain, dtest, droped_features):
    gbdt_model = xgb.train(
        param,
        dtrain,
        evals=[(dtest, "test"), (dtrain, "train")],
        verbose_eval=50,
        early_stopping_rounds=10,
        num_boost_round=steps,
    )
    return gbdt_model


def sample_noisy_feature(point, train_df, model, dropped_feature):
    sum = 0
    count = 0
    for _ in range(0, 500):
        point_cp = point.copy()
        for name in dropped_feature:
            X = train_df[name].values
            index = np.random.randint(0, len(X))
            point_cp[name] = X[index]
        sum += model.predict(xgb.DMatrix(point_cp))[0]
        count += 1
    return sum / count


def get_shap_ranking(model, data):
    sh = ShapWrapper()
    results = sh.get_shap_ranking(model, data)
    return results


def run_features_noising(
    param,
    steps,
    train_path,
    test_path,
    stddev,
    model_path,
    target_column,
    perturbed,
    point_index=None,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")
    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    model_xgb = xgb.Booster()
    model_xgb.load_model(model_path)

    cpp_tree = TreeWrapper(model_name, model_dir_path)
    if point_index is None:
        point_index = np.random.randint(0, len(test_df))

    p = test_df.loc[point_index, test_df.columns != target_column]
    eval = cpp_tree.eval(p)
    print(f"True: {test_df.loc[point_index, :][target_column]}")
    print(f"Predicted: {eval}")
    print(f"Bias: {cpp_tree.bias}")

    print(
        f"Error: {test_df.loc[point_index, test_df.columns == target_column] - eval }"
    )
    ranking = cpp_tree.rank_features(p, stddev)
    X = test_df.iloc[[point_index], test_df.columns != target_column]
    shap_ranking = get_shap_ranking(model_xgb, X)[0]  # [point_index]
    print(shap_ranking)
    point_x = train_df.iloc[[point_index], train_df.columns != target_column]
    point_y = train_df.iloc[point_index, train_df.columns == target_column]
    print(point_y, point_x)
    # dpoint = xgb.DMatrix(point_x, label=point_y)

    ev = sample_noisy_feature(
        point_x,
        train_df,
        model_xgb,
        ranking[0:perturbed],
    )

    print(ev)
    print(f"Predicted: {eval  }")
    print(test_df.loc[point_index, :][target_column])

    point_x = train_df.iloc[[point_index], train_df.columns != target_column]
    ev_shap = sample_noisy_feature(
        point_x,
        train_df,
        model_xgb,
        shap_ranking[0:perturbed],
    )

    print("SHAP ", ev_shap)
    print(f"Predicted: {eval  }")
    print(test_df.loc[point_index, :][target_column])

    return [ev, ev_shap, eval, test_df.loc[point_index, :][target_column]]


def run_retraining_experiment(
    param,
    steps,
    train_path,
    test_path,
    stddev,
    model_path,
    target_column,
    perturbed=1,
    point_index=None,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.loc[:, train_df.columns != target_column]
    y_train = train_df.loc[:, train_df.columns == target_column]

    X_test = test_df.loc[:, test_df.columns != target_column]
    y_test = test_df.loc[:, test_df.columns == target_column]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")
    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    model_xgb = xgb.Booster()
    model_xgb.load_model(model_path)

    cpp_tree = TreeWrapper(model_name, model_dir_path)
    if point_index is None:
        point_index = np.random.randint(0, len(test_df))

    p = test_df.loc[point_index, test_df.columns != target_column]
    eval = cpp_tree.eval(p)
    print(f"True: {test_df.loc[point_index, :][target_column]}")
    print(f"Predicted: {eval  }")
    print(f"Bias: {cpp_tree.bias}")

    print(
        f"Error: {test_df.loc[point_index, test_df.columns == target_column] - eval }"
    )
    ranking = cpp_tree.rank_features(p, stddev)
    X = test_df.iloc[:, test_df.columns != target_column]
    shap_ranking = get_shap_ranking(model_xgb, X)[point_index]
    print(shap_ranking)
    dtrain = xgb.DMatrix(X_train.drop(columns=ranking[0:perturbed]), label=y_train)
    dtest = xgb.DMatrix(X_test.drop(columns=ranking[0:perturbed]), label=y_test)
    model = train_and_save_model(param, steps, dtrain, dtest, "")
    ev = model.predict(dtest)[point_index]

    print(ev)
    print(f"Predicted: {eval  }")
    print(test_df.loc[point_index, :][target_column])

    dtrain = xgb.DMatrix(X_train.drop(columns=shap_ranking[0:perturbed]), label=y_train)
    dtest = xgb.DMatrix(X_test.drop(columns=shap_ranking[0:perturbed]), label=y_test)
    model = train_and_save_model(param, steps, dtrain, dtest, "")
    ev_shap = model.predict(dtest)[point_index]

    print("SHAP ", ev_shap)
    print(f"Predicted: {eval  }")
    print(test_df.loc[point_index, :][target_column])
    return [ev, ev_shap, eval, test_df.loc[point_index, :][target_column]]


def run_exp_for_wine(std, perturbed):
    param = {
        "eta": 0.2,
        "max_depth": 4,
        "objective": "reg:squarederror",
        "seed": 42,
        "subsample": 0.8,
    }
    steps = 32
    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []
    true_pred_list = []
    df = pd.read_csv("data/wine_quality/test_winequality_red_scaled.csv")
    test_len = len(df)
    for i in range(0, test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_features_noising(
            param,
            steps,
            "data/wine_quality/train_winequality_red_scaled.csv",
            "data/wine_quality/test_winequality_red_scaled.csv",
            std,
            "models/winequality_red_saved.json",
            target_column="quality",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/wine_model/ranking_comparision/ranking_comparision_noising_std_{std}_tested_features_{perturbed}",
        arr,
    )

    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []
    true_pred_list = []

    for i in range(0, test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_retraining_experiment(
            param,
            steps,
            "data/wine_quality/train_winequality_red_scaled.csv",
            "data/wine_quality/test_winequality_red_scaled.csv",
            std,
            "models/winequality_red_saved.json",
            target_column="quality",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/wine_model/ranking_comparision/ranking_comparision_retraining_std_{std}_tested_features_{perturbed}",
        arr,
    )


def run_exp_for_housing(std, perturbed):
    # training hyperparameters
    param = {
        "eta": 0.1,
        "max_depth": 4,
        "objective": "reg:squarederror",
        "seed": 42,
        "subsample": 0.01,
    }
    steps = 40
    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []
    df = pd.read_csv("data/housing_data/test_housing_scaled.csv")
    test_len = len(df)
    true_pred_list = []
    for i in range(0, test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_features_noising(
            param=param,
            steps=steps,
            train_path="data/housing_data/train_housing_scaled.csv",
            test_path="data/housing_data/test_housing_scaled.csv",
            stddev=std,
            model_path="models/housing_saved.json",
            target_column="median_house_value",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/housing_model/ranking_comparision/ranking_comparision_noising_std_{std}_tested_features_{perturbed}",
        arr,
    )

    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []
    true_pred_list = []
    for i in range(0, test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_retraining_experiment(
            param=param,
            steps=steps,
            train_path="data/housing_data/train_housing_scaled.csv",
            test_path="data/housing_data/test_housing_scaled.csv",
            stddev=std,
            model_path="models/housing_saved.json",
            target_column="median_house_value",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/housing_model/ranking_comparision/ranking_comparision_retraining_std_{std}_tested_features_{perturbed}",
        arr,
    )


def run_exp_for_telemetry(std, perturbed):
    # training hyperparameters
    param = {
        "eta": 0.2,
        "max_depth": 4,
        "objective": "reg:squarederror",
        "seed": 42,
        "subsample": 0.8,
    }
    steps = 40
    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []

    df = pd.read_csv("data/telemetry/test_telemetry_scaled.csv")
    test_len = len(df)
    true_pred_list = []
    for i in range(0, 2):#test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_features_noising(
            param=param,
            steps=steps,
            train_path="data/telemetry/train_telemetry_scaled.csv",
            test_path="data/telemetry/test_telemetry_scaled.csv",
            stddev=std,
            model_path="models/telemetry_saved.json",
            target_column="motor_UPDRS",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/telemetry_model/ranking_comparision/ranking_comparision_noising_std_{std}_tested_features_{perturbed}",
        arr,
    )

    pg_pred_list = []
    shap_pred_list = []
    orig_pred_list = []
    true_pred_list = []
    for i in range(0, 2):#test_len):
        pg_pred, shap_pred, orig_pred, true_pred = run_retraining_experiment(
            param=param,
            steps=steps,
            train_path="data/telemetry/train_telemetry_scaled.csv",
            test_path="data/telemetry/test_telemetry_scaled.csv",
            stddev=std,
            model_path="models/telemetry_saved.json",
            target_column="motor_UPDRS",
            perturbed=perturbed,
            point_index=i,
        )
        pg_pred_list.append(pg_pred)
        shap_pred_list.append(shap_pred)
        orig_pred_list.append(orig_pred)
        true_pred_list.append(true_pred)

    results = [pg_pred_list, shap_pred_list, orig_pred_list, true_pred_list]
    arr = np.array(results)
    print(arr)
    np.save(
        f"results/telemetry_model/ranking_comparision/ranking_comparision_retraining_std_{std}_tested_features_{perturbed}",
        arr,
    )


if __name__ == "__main__":
    for f in [1]:
        for s in [0.1]:
            run_exp_for_telemetry(s, f)
            run_exp_for_housing(s, f)
