import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    rank_features_by_random,
)
from src.decision_tree.tree import load_trees

while "notebooks" in os.getcwd():
    os.chdir("../")


def rank_features_timer_wrapper(*args):
    t = time.time()
    predgap.rank_features(*args)
    t = time.time() - t
    return t


def rank_features_by_random_timer_wrapper(*args):
    t = time.time()
    print("ARGS", args)
    rank_features_by_random(*args)
    t = time.time() - t
    return t


def calculate_rankings(
    stddev: float, proc_number: int, num_iter: list[int], number_of_samples: int
):
    models_path = Path("models")
    data_path = Path("data")
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)
    global predgap
    predgap = NormalPredictionGap(stddev)

    model_xgb = xgb.Booster()
    model_xgb.load_model(models_path / (wine_model_name + "_saved.json"))

    random_points = random.sample(range(0, len(wine_data)), number_of_samples)

    points = []
    for i in random_points:
        points.append((wine_trees, wine_data.iloc[i, :-1]))
    pool = Pool(processes=proc_number)
    results = np.array(pool.starmap(rank_features_timer_wrapper, points))
    exact_mean_time = np.mean(results)

    random_mean_times = []
    for n in num_iter:
        points = []
        for i in random_points:
            points.append((wine_trees, wine_data.iloc[i, :-1], stddev, n))
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(rank_features_by_random_timer_wrapper, points))
        random_mean_times.append(np.mean(results))
    diffs = [np.abs(i - exact_mean_time) for i in random_mean_times]
    print(f"Best number of iterations: {num_iter[diffs.index(min(diffs))]}")


if __name__ == "__main__":
    stdev = 0.3
    proc_num = 20
    num_iters = [3000, 3500, 4000, 4500]
    number_of_samples = 20
    calculate_rankings(stdev, proc_num, num_iters, number_of_samples)
