import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from multiprocessing import Pool

from src.decision_tree.tree import load_trees
from src.decision_tree.prediction_gap import (
    prediction_gap_by_exact_calc_single_datapoint,
    prediction_gap_by_random_sampling_single_datapoint,
    NormalPredictionGap,
)
import random


while "notebooks" in os.getcwd():
    os.chdir("../")


def test_differences_between_approx_and_exact(stddev: float, iterations: int):
    models_path = Path("models")
    data_path = Path("data")
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)

    model_xgb = xgb.Booster()
    model_xgb.load_model(models_path / (wine_model_name + "_saved.json"))

    predgap = NormalPredictionGap(stddev)

    random_point = random.sample(range(0, len(wine_data)), 1)[0]
    all_features = list(wine_data.columns.values)
    random_features = random.sample(all_features, random.randint(0, len(all_features)))
    tmp = prediction_gap_by_random_sampling_single_datapoint(
        trees=wine_trees,
        data_point=wine_data.iloc[random_point, :],
        perturbed_features=set(random_features),
        stddev=stddev,
        squared=True,
        num_iter=iterations,
    )

    tmp2 = prediction_gap_by_exact_calc_single_datapoint(
        predgap=predgap,
        trees=wine_trees,
        data=wine_data.iloc[[random_point], :],
        perturbed_features=set(random_features),
        squared=True,
    )
    return [tmp, tmp2[0], len(random_features)]


def run_experiment(
    iterations: int, samples: int, stddev: float, results_path: Path, proc_number: int
):
    args = []
    for _ in range(0, samples):
        args.append((stddev, iterations))

    pool = Pool(processes=proc_number)
    results = np.array(pool.starmap(test_differences_between_approx_and_exact, args))
    print(results)
    np.save(
        (
            results_path
            / f"precision_comparision_std_{stddev}_iterations_{iterations}.npy"
        ),
        results,
    )


if __name__ == "__main__":
    results_path = Path("results/precision/")
    proc_number = 30
    stdev = 0.3
    iterations = [500, 1000, 2000, 4000, 8000, 10000]
    models_path = Path("models")
    for i in iterations:
        run_experiment(i, 200, stdev, results_path, proc_number)
