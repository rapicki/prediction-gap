from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.decision_tree.tree import load_trees


def get_models_dirs(model_path: str, data_path: str, results_path: str):
    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")

    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    wine_trees = load_trees(model_dir_path, model_name)
    wine_data = pd.read_csv(data_path)

    results_path = Path(results_path)

    model_xgb = xgb.Booster()
    model_xgb.load_model(model_path)

    return wine_trees, wine_data, results_path, model_xgb
