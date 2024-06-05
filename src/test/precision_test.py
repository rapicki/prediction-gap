import time
from pathlib import Path

import pandas as pd
from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.decision_tree.tree import TreeEnsemble, parse_xgboost_dump
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper


def test_trees(models_path: Path, model_name: str, df: pd.DataFrame):
    cpp_tree = TreeWrapper(model_name, models_path)
    print()
    df_ = df.iloc[0, :]
    array = df_.to_numpy()
    names = list(df_.keys().values)
    s = cpp_tree.t.expected_diff_squared(
        array,
        names,
        ["housing_median_age"],
        1.0,
    )

    print("C++ pred_gap: ", s)


if __name__ == "__main__":
    model_path = Path("./models")
    model_name = "winequality_red_single"
    wine_data = pd.read_csv(Path("./data/wine_quality/winequality_red_scaled.csv"))
    test_trees(model_path, model_name, wine_data)

    model_path = Path("./models")
    model_name = "housing"
    wine_data = pd.read_csv(Path("./data/housing_data/test_housing_scaled.csv"))
    test_trees(model_path, model_name, wine_data)
