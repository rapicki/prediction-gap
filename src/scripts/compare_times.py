import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    prediction_gap_by_exact_calc_single_datapoint,
    prediction_gap_by_random_sampling_single_datapoint,
)
from src.decision_tree.tree import load_trees

while "notebooks" in os.getcwd():
    os.chdir("../")


def test_differences_between_approx_and_exact(
    stddev: float, iterations: int, point_ind: int, random_features: list[int]
):
    models_path = Path("models")
    data_path = Path("data")
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)

    model_xgb = xgb.Booster()
    model_xgb.load_model(models_path / (wine_model_name + "_saved.json"))

    predgap = NormalPredictionGap(stddev)

    random_point = wine_data.iloc[[point_ind], :]

    tmp = prediction_gap_by_random_sampling_single_datapoint(
        trees=wine_trees,
        data_point=random_point,
        perturbed_features=set(random_features),
        stddev=stddev,
        squared=True,
        num_iter=iterations,
    )

    tmp2 = prediction_gap_by_exact_calc_single_datapoint(
        predgap=predgap,
        trees=wine_trees,
        data=random_point,
        perturbed_features=set(random_features),
        squared=True,
    )
    return [tmp, tmp2[0], len(random_features)]


def sample_indices_and_subsets(number: int):
    data_path = Path("data")
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_data = pd.read_csv(wine_test_data_path)

    all_features = list(wine_data.columns.values)[:-1]
    samples = []
    for _ in range(number):
        random_features = random.sample(
            all_features, random.randint(1, len(all_features))
        )
        random_point = random.sample(range(0, len(wine_data)), 1)[0]
        samples.append((random_features, random_point))
    return samples


def run_experiment(
    iterations: int, stddev: float, results_path: Path, proc_number: int, samples: list
):
    args = []
    for subset, point in samples:
        args.append((stddev, iterations, point, subset))

    pool = Pool(processes=proc_number)
    results = np.array(pool.starmap(test_differences_between_approx_and_exact, args))
    np.save(
        (
            results_path
            / f"precision_comparision_std_{stddev}_iterations_{iterations}.npy"
        ),
        results,
    )


if __name__ == "__main__":
    results_path = Path("results/precision/")
    proc_number = 10
    stdev = 0.3
    iterations = [10, 50, 100, 500, 1000, 2000, 4000, 8000, 10000]
    models_path = Path("models")
    samples = sample_indices_and_subsets(10000)
    for i in iterations:
        run_experiment(i, stdev, results_path, proc_number, samples)
