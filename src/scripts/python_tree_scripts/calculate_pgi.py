import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.scripts.python_tree_scripts.utils import get_models_dirs

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
    predgap = PerturbPredictionGap(stddev)

    wine_trees, wine_data, results_path, model_xgb = get_models_dirs(
        model_path, data_path, results_path
    )

    assert len(wine_data.iloc[0]) - 1 >= top_k

    for pgi in pgi_names:
        ranking = np.load(ranking_file)
        points = []
        for i in range(0, len(wine_data)):
            data_point = wine_data.iloc[i, :]
            baseline_pred = wine_trees.eval(data_point)

            points.append(
                (
                    wine_trees,
                    wine_data.iloc[i, :-1],
                    set(ranking[i].tolist()[0:top_k]),
                    baseline_pred,
                )
            )
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(predgap.prediction_gap_fixed, points))
        np.save((results_path / pgi), results)


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


"""if __name__ == "__main__":
    proc_num = 4
    stddevs = [0.01, 0.03, 0.1, 0.3]
    ### calculating pgi for shap ranking
    names = []
    for s1 in stddevs:
        s_names = []
        for k in range(1, 12):
            ranking_names = []
            pgi_file_names = []
            ranking_name = "wine_shap_ranking.npy"
            pgi_file_name = f"pgi_wine_shap_ranking_pgi_std_{s1}_topk_{k}.npy"
            ranking_names.append(ranking_name)
            pgi_file_names.append(pgi_file_name)
            s_names.append([ranking_names, pgi_file_names])
        names.append(s_names)

    for s, name in zip(stddevs, names):
        for k, n in zip(range(1, 12), name):
            print(n)
            calculate_rankings(s, proc_num, n[0], n[1], top_k=k)

    ### calculating pgi for exact ranking
    names = []
    for s1 in stddevs:
        s_names = []
        for k in range(1, 12):
            ranking_names = []
            pgi_file_names = []
            ranking_name = f"wine_exact_ranking_std_{s1}.npy"
            pgi_file_name = f"pgi_wine_exact_ranking_std_{s1}_pgi_std_{s1}_topk_{k}.npy"
            ranking_names.append(ranking_name)
            pgi_file_names.append(pgi_file_name)
            s_names.append([ranking_names, pgi_file_names])
        names.append(s_names)

    for s, name in zip(stddevs, names):
        for k, n in zip(range(1, 12), name):
            print(n)
            calculate_rankings(s, proc_num, n[0], n[1], top_k=k)

    ### calculating pgi for approx ranking
    names = []
    for s1 in stddevs:
        s_names = []
        for k in range(1, 12):
            ranking_names = []
            pgi_file_names = []
            ranking_name = "wine_approx_ranking_4000_iter.npy"
            pgi_file_name = f"pgi_wine_approx_ranking_4000_pgi_std_{s1}_topk_{k}.npy"
            ranking_names.append(ranking_name)
            pgi_file_names.append(pgi_file_name)
            s_names.append([ranking_names, pgi_file_names])
        names.append(s_names)

    for s, name in zip(stddevs, names):
        for k, n in zip(range(1, 12), name):
            print(n)
            calculate_rankings(s, proc_num, n[0], n[1], top_k=k)
"""
