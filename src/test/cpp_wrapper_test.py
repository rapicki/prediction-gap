from src.tree_cpp.cpp_tree_wrapper import TreeWrapper
from src.decision_tree.tree import parse_xgboost_dump, TreeEnsemble
import pandas as pd
from pathlib import Path


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


if __name__ == "__main__":
    model_path = Path("./models")
    model_name = "winequality_red"
    wine_data = pd.read_csv(Path("./data/wine_quality/winequality_red_scaled.csv"))
    test_trees(model_path, model_name, wine_data)

    model_path = Path("./models")
    model_name = "housing"
    wine_data = pd.read_csv(Path("./data/housing_data/housing_scaled.csv"))
    test_trees(model_path, model_name, wine_data)
