import os
from multiprocessing import Pool

import numpy as np

from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    rank_features_by_random,
)
from src.decision_tree.shap_wrapper import ShapWrapper
from src.scripts.python_tree_scripts.utils import get_models_dirs

while "notebooks" in os.getcwd():
    os.chdir("../")


def calculate_rankings(
    stddev: float,
    proc_number: int,
    model_path: str,
    data_path: str,
    results_path: str,
    name: str,
    ranking_types: list[str] = ["exact", "approx", "shap"],
    num_iter: int = 100,
):
    predgap = NormalPredictionGap(stddev)
    wine_trees, wine_data, results_path, model_xgb = get_models_dirs(
        model_path, data_path, results_path
    )

    if "exact" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_trees, wine_data.iloc[i, :-1]))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(predgap.rank_features, points))
        np.save(results_path / (f"{name}_exact_ranking_std_{stddev}.npy"), results)

    if "approx" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_trees, wine_data.iloc[i, :-1], stddev, num_iter))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(rank_features_by_random, points))
        np.save(
            results_path
            / (f"{name}_approx_ranking_{num_iter}_iter_{stddev}_stddev.npy"),
            results,
        )

    if "shap" in ranking_types:
        sh = ShapWrapper()
        if "quality" in wine_data.columns:
            X = wine_data.loc[:, wine_data.columns != "quality"]
        elif "median_house_value":
            X = wine_data.loc[:, wine_data.columns != "median_house_value"]

        results = sh.get_shap_ranking(model_xgb, X)
        np.save(results_path / (f"{name}_shap_ranking.npy"), results)


def calculate_ranking_main(args):
    stddev = [float(i) for i in args.stddev_list.split(",")]
    iterations = args.iterations
    result_path = args.results_dir
    proc_num = args.proc_num
    ranking_methods = args.ranking_method
    data_path = args.data
    model_path = args.model
    name = args.data_name
    print(stddev, iterations)
    for dev in stddev:
        calculate_rankings(
            dev,
            proc_num,
            model_path,
            data_path,
            result_path,
            name,
            [ranking_methods],
            iterations,
        )


"""
if __name__ == "__main__":
    stdev = [0.01, 0.03, 0.1, 0.3]
    proc_num = 2
    num_iter = 4000
    ranking_type = ["exact"]
    for i in stdev:
        calculate_rankings(i, proc_num, num_iter, ranking_type)
"""
