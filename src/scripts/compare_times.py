import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    prediction_gap_by_exact_calc_single_datapoint,
    prediction_gap_by_random_sampling_single_datapoint,
)
from src.scripts.utils import get_models_dirs

while "notebooks" in os.getcwd():
    os.chdir("../")


def test_differences_between_approx_and_exact(
    stddev: float,
    model_path: str,
    data_path: str,
    results_path: str,
    iterations: int,
    point_ind: int,
    random_features: list[int],
):
    wine_trees, wine_data, results_path, model_xgb = get_models_dirs(
        model_path, data_path, results_path
    )

    predgap = NormalPredictionGap(stddev)

    random_point = wine_data.iloc[[point_ind], :]
    t1 = time.time()
    tmp = prediction_gap_by_random_sampling_single_datapoint(
        trees=wine_trees,
        data_point=random_point,
        perturbed_features=set(random_features),
        stddev=stddev,
        squared=True,
        num_iter=iterations,
    )
    t1 = time.time() - t1

    t2 = time.time()
    tmp2 = prediction_gap_by_exact_calc_single_datapoint(
        predgap=predgap,
        trees=wine_trees,
        data=random_point,
        perturbed_features=set(random_features),
        squared=True,
    )
    t2 = time.time() - t2
    return [tmp, tmp2[0], len(random_features), t1, t2]


def sample_indices_and_subsets(number: int, file: str):
    wine_data = pd.read_csv(file)

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
    iterations: int,
    stddev: float,
    results_path: str,
    model_path: str,
    data_path: str,
    proc_number: int,
    samples: list,
):
    args = []
    for subset, point in samples:
        args.append(
            (stddev, model_path, data_path, results_path, iterations, point, subset)
        )
    pool = Pool(processes=proc_number)
    results = np.array(pool.starmap(test_differences_between_approx_and_exact, args))
    results_path = Path(results_path)
    np.save(
        (
            results_path
            / f"precision_comparision_std_{stddev}_iterations_{iterations}.npy"
        ),
        results,
    )


def compare_times_main(args):
    stddevs = [float(i) for i in args.stddev_list.split(",")]
    iterations = [int(i) for i in args.iterations_list.split(",")]
    result_path = args.results_dir
    proc_num = args.proc_num
    data_path = args.data
    model_path = args.model
    samples = args.samples
    samples = sample_indices_and_subsets(samples, data_path)
    for s in stddevs:
        for i in iterations:
            run_experiment(i, s, result_path, model_path, data_path, proc_num, samples)


"""if __name__ == "__main__":
    results_path = Path("results/precision/")
    proc_number = 10
    stdev = 0.3
    iterations = [10, 50, 100, 500, 1000, 2000, 4000, 8000, 10000]
    models_path = Path("models")
    samples = sample_indices_and_subsets(10000)
    for i in iterations:
        run_experiment(i, stdev, results_path, proc_number, samples)"""
