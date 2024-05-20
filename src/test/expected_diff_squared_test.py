from pathlib import Path

import pandas as pd
from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.decision_tree.tree import TreeEnsemble, parse_xgboost_dump
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper

import time


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
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality",
        ],
        1.0,
    )

    print("C++ pred_gap: ", s, f", time: {time.time() - t}")

    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    python_tree = TreeEnsemble(
        trees_from_dump.trees, models_path / f"{model_name}_saved.json"
    )
    predgap = PerturbPredictionGap(1.0)
    t = time.time()
    p_eval = python_tree.eval(df.iloc[0, :])
    s = predgap.prediction_gap_fixed(
        python_tree,
        wine_data.iloc[0, :],
        [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality",
        ],
        p_eval,
    )
    print("Python pred_gap: ", s, f", time: {time.time() - t}")


if __name__ == "__main__":
    model_path = Path("./models")
    model_name = "winequality_red"
    wine_data = pd.read_csv(Path("./data/wine_quality/winequality_red_scaled.csv"))
    test_trees(model_path, model_name, wine_data)
