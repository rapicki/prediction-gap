import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    rank_features_by_random,
)
from src.decision_tree.shap_wrapper import ShapWrapper
from src.decision_tree.tree import load_trees

while "notebooks" in os.getcwd():
    os.chdir("../")


def calculate_rankings(
    stddev: float,
    proc_number: int,
    num_iter: int = 100,
    ranking_types: list[str] = ["exact", "approx", "shap"],
):
    models_path = Path("models")
    data_path = Path("data")
    results_path = Path("results")
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)
    predgap = NormalPredictionGap(stddev)

    model_xgb = xgb.Booster()
    model_xgb.load_model(models_path / (wine_model_name + "_saved.json"))

    if "exact" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_trees, wine_data.iloc[i, :-1]))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(predgap.rank_features, points))
        np.save(results_path / (f'wine" + "_exact_ranking_std_{stddev}.npy'), results)
    if "approx" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_trees, wine_data.iloc[i, :-1], stddev, num_iter))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(rank_features_by_random, points))
        np.save(
            results_path / ("wine" + f"_approx_ranking_{num_iter}_iter.npy"), results
        )
    if "approx" in ranking_types:
        sh = ShapWrapper()
        X = wine_data.loc[:, wine_data.columns != "quality"]
        results = sh.get_shap_ranking(model_xgb, X)
        np.save(results_path / ("wine" + "_shap_ranking.npy"), results)


if __name__ == "__main__":
    stdev = [0.01, 0.03, 0.1, 0.3]
    proc_num = 2
    num_iter = 4000
    ranking_type = ["exact"]
    for i in stdev:
        calculate_rankings(i, proc_num, num_iter, ranking_type)
