import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from src.decision_tree.prediction_gap import PerturbPredictionGap
from src.decision_tree.tree import load_trees

while "notebooks" in os.getcwd():
    os.chdir("../")


def calculate_rankings(
    stddev: float,
    proc_number: int,
    ranking_names: list[str],
    pgi_names: list[str],
    top_k: int = 11,
):
    models_path = Path("models")
    data_path = Path("data")
    results_path = Path("results")
    save_path = Path("results/top_k_pgi")

    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)

    predgap = PerturbPredictionGap(stddev)
    wine_trees = load_trees(models_path, wine_model_name)

    assert len(wine_data.iloc[0]) - 1 >= top_k

    for rnk, pgi in zip(ranking_names, pgi_names):
        ranking = np.load(results_path / rnk)
        points = []
        for i in range(0, 2):  # len(wine_data)):
            data_point = wine_data.iloc[i, :]
            baseline_pred = wine_trees.eval(data_point)

            points.append(
                (
                    wine_trees,
                    wine_data.iloc[i, :-1],
                    baseline_pred,
                    ranking[i].tolist()[0:top_k],
                )
            )
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(predgap.pgi, points))
        np.save((save_path / pgi), results)


if __name__ == "__main__":
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
