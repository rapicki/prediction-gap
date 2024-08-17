import os
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool

import numpy as np

from src.decision_tree.shap_wrapper import ShapWrapper
from src.scripts.python_tree_scripts.utils import get_models_dirs
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper
from pathlib import Path
import pandas as pd

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
    wine_trees, wine_data, results_path, model_xgb = get_models_dirs(
        model_path, data_path, results_path
    )
    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")
    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    wine_data = pd.read_csv(data_path)

    cpp_tree = TreeWrapper(model_name, model_dir_path)

    if "exact" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_data.iloc[i, :-1], stddev))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(cpp_tree.rank_features, points))
        np.save(results_path / (f"{name}_exact_ranking_std_{stddev}.npy"), results)

    if "approx" in ranking_types:
        points = []
        for i in range(0, len(wine_data)):
            points.append((wine_data.iloc[i, :-1], stddev, num_iter))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(cpp_tree.rank_features_by_random, points))
        np.save(
            results_path
            / (f"{name}_approx_ranking_{num_iter}_iter_{stddev}_stddev.npy"),
            results,
        )

    if "shap" in ranking_types:
        sh = ShapWrapper()
        if "quality" in wine_data.columns:
            X = wine_data.loc[:, wine_data.columns != "quality"]
        elif "median_house_value" in wine_data.columns:
            X = wine_data.loc[:, wine_data.columns != "median_house_value"]
        elif "motor_UPDRS" in wine_data.columns:
            X = wine_data.loc[:, wine_data.columns != "motor_UPDRS"]

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
