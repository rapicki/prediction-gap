from pathlib import Path
import time

import pandas as pd
from src.decision_tree.tree import TreeEnsemble, parse_xgboost_dump
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper


def load_trees(models_path: Path, model_name: str, df: pd.Series):
    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    t = TreeEnsemble(trees_from_dump.trees, models_path / f"{model_name}_saved.json")
    return t.eval(df)


def test_trees(models_path: Path, model_name: str, df: pd.DataFrame):
    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    python_tree = TreeEnsemble(
        trees_from_dump.trees, models_path / f"{model_name}_saved.json"
    )
    cpp_tree = TreeWrapper(model_name, models_path)
    for i in range(len(df)):
        p_eval = python_tree.eval(df.iloc[i, :])
        cpp_eval = cpp_tree.eval(df.iloc[i, :])
        print(p_eval, cpp_eval)
        assert p_eval == cpp_eval


def test_eval_times(models_path: Path, model_name: str, df: pd.DataFrame):
    trees_from_dump = parse_xgboost_dump(models_path / f"{model_name}_dumped.txt")
    python_tree = TreeEnsemble(
        trees_from_dump.trees, models_path / f"{model_name}_saved.json"
    )
    cpp_tree = TreeWrapper(model_name, models_path)

    t = time.time()
    for i in range(1000):
        p_eval = python_tree.eval(df.iloc[i, :])

    print(f"Python time for one eval: {(time.time() - t)/1000}")

    t = time.time()
    for i in range(1000):
        cpp_eval = cpp_tree.eval(df.iloc[i, :])

    print(f"C++ time for one eval: {(time.time() - t)/1000}")


if __name__ == "__main__":
    model_path = Path("./models")
    model_name = "winequality_red"
    wine_data = pd.read_csv(Path("./data/wine_quality/winequality_red_scaled.csv"))
    test_trees(model_path, model_name, wine_data)
    test_eval_times(model_path, model_name, wine_data)

    model_path = Path("./models")
    model_name = "housing"
    wine_data = pd.read_csv(Path("./data/housing_data/housing_scaled.csv"))
    #test_trees(model_path, model_name, wine_data)
