import time
from pathlib import Path

import pandas as pd
from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.decision_tree.tree import TreeEnsemble, parse_xgboost_dump
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper


def load_trees(models_path: Path, model_name: str, df: pd.Series):
    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    t = TreeEnsemble(trees_from_dump.trees, models_path / f"{model_name}_saved.json")
    return t.eval(df)


def test_trees(models_path: Path, model_name: str, df: pd.DataFrame):
    cpp_tree = TreeWrapper(model_name, models_path)
    cpp_eval = cpp_tree.eval(df.iloc[0, :])
    print(cpp_eval)
    df_ = df.iloc[0, :]
    array = df_.to_numpy()
    names = list(df_.keys().values)
    t = time.time()
    s = cpp_tree.t.expected_diff_squared(
        array,
        names,
        [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
        ],
        0.3,
    )

    print("C++ pred_gap: ", s, f", time: {time.time() - t}")

    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    python_tree = TreeEnsemble(
        trees_from_dump.trees, models_path / f"{model_name}_saved.json"
    )
    predgap = PerturbPredictionGap(0.3)
    t = time.time()
    p_eval = python_tree.eval(df.iloc[0, :])
    s = predgap.prediction_gap_fixed(
        python_tree,
        wine_data.iloc[0, :],
        [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
        ],
        p_eval,
    )
    print("Python pred_gap: ", s, f", time: {time.time() - t}")


def test_predgap_times(models_path: Path, model_name: str, df: pd.DataFrame):
    cpp_tree = TreeWrapper(model_name, models_path)
    t = time.time()
    for i in range(1):
        df_ = df.iloc[i, :]
        array = df_.to_numpy()
        names = list(df_.keys().values)
        s = cpp_tree.t.expected_diff_squared(
            array,
            names,
            [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
            ],
            1.0,
        )

    print("C++ pred_gap: ", s, f", time: {(time.time() - t)/10}")

    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    python_tree = TreeEnsemble(
        trees_from_dump.trees, models_path / f"{model_name}_saved.json"
    )
    predgap = PerturbPredictionGap(1.0)
    t = time.time()
    for i in range(0):
        p_eval = python_tree.eval(df.iloc[i, :])
        s = predgap.prediction_gap_fixed(
            python_tree,
            wine_data.iloc[i, :],
            [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
            ],
            p_eval,
        )
    # print("Python pred_gap: ", s, f", time: {(time.time() - t)/10}")


def test_fast_eval(models_path: Path, model_name: str, df: pd.DataFrame):
    cpp_tree = TreeWrapper(model_name, models_path)
    t = time.time()
    for i in range(1):
        df_ = df.iloc[i, :]
        s = cpp_tree.prediction_gap_sampling_fast(
            df_,
            {
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
            },
            1.0,
            num_iter=100,
        )

    print("C++ pred_gap_monte: ", s, f", time: {(time.time() - t)/10}")

    cpp_tree = TreeWrapper(model_name, models_path)
    t = time.time()
    for i in range(1):
        df_ = df.iloc[i, :]
        s = cpp_tree.prediction_gap_sampling_fast_quasi(
            df_,
            {
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
            },
            1.0,
            num_iter=3000000,
        )

    print("C++ pred_gap_quasi: ", s, f", time: {(time.time() - t)/10}")


if __name__ == "__main__":
    model_path = Path("./models")
    model_name = "housing_single"
    wine_data = pd.read_csv(Path("./data/housing_data/test_housing_scaled.csv"))
    # test_trees(model_path, model_name, wine_data)
    test_predgap_times(model_path, model_name, wine_data)
    test_fast_eval(model_path, model_name, wine_data)
