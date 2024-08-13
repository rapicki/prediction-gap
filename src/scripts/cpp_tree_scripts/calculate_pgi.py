import os
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path

import numpy as np
import pandas as pd
from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.scripts.python_tree_scripts.utils import get_models_dirs
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper

while "notebooks" in os.getcwd():
    os.chdir("../")


def calculate_rankings(
    stddev: float,
    proc_number: int,
    ranking_file: str,
    pgi_names: list[str],
    model_path: str,
    data_path: str,
    results_path: str,
    top_k: int = 11,
):
    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")
    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    wine_data = pd.read_csv(data_path)

    cpp_tree = TreeWrapper(model_name, model_dir_path)

    assert len(wine_data.iloc[0]) - 1 >= top_k

    for pgi in pgi_names:
        ranking = np.load(ranking_file)
        points = []
        for i in range(0, len(wine_data)):
            points.append(
                (wine_data.iloc[i, :-1], set(ranking[i].tolist()[0:top_k]), stddev)
            )
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(cpp_tree.prediction_gap_exact, points))
        np.save((results_path +  pgi), results)


def calculate_pgi_main(args):
    stddev = args.stddev
    result_path = args.results_dir
    proc_num = args.proc_num
    data_path = args.data
    model_path = args.model
    features_number = args.features_number
    ranking_file = args.ranking_file

    slash_indx = ranking_file.rfind("/")
    dot_indx = ranking_file.rfind(".")
    ranking_name = ranking_file[slash_indx + 1 : dot_indx]
    name = []
    pgi_path = Path(result_path) / ranking_name
    pgi_path.mkdir(parents=True, exist_ok=True)
    for k in range(1, features_number + 1):
        pgi_file_names = []
        pgi_file_name = f"{ranking_name}/pgi_std_{stddev}_topk_{k}.npy"
        pgi_file_names.append(pgi_file_name)
        name.append(pgi_file_names)

    for k, n in zip(range(1, features_number + 1), name):
        calculate_rankings(
            stddev,
            proc_num,
            ranking_file,
            n,
            model_path,
            data_path,
            result_path,
            top_k=k,
        )
