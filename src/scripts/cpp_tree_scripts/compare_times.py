import os
import pickle
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from src.tree_cpp.cpp_tree_wrapper import TreeWrapper

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
    slash_indx = model_path.rfind("/")
    dot_indx = model_path.rfind("_")
    model_dir_path = Path(model_path[0:slash_indx])
    model_name = model_path[slash_indx + 1 : dot_indx]

    wine_data = pd.read_csv(data_path)

    cpp_tree = TreeWrapper(model_name, model_dir_path)

    random_point = wine_data.iloc[point_ind, :]
    t1 = time.time()
    tmp = cpp_tree.prediction_gap_sampling_fast(
        data_point=random_point,
        perturbed_features=set(random_features),
        stddev=stddev,
        num_iter=iterations,
    )
    t1 = time.time() - t1

    t2 = time.time()
    tmp2 = cpp_tree.prediction_gap_exact(
        data_point=random_point,
        perturbed_features=set(random_features),
        stddev=stddev,
    )
    t2 = time.time() - t2

    t3 = time.time()
    tmp3 = cpp_tree.prediction_gap_sampling_fast_quasi(
        data_point=random_point,
        perturbed_features=set(random_features),
        stddev=stddev,
        num_iter=iterations,
    )
    t3 = time.time() - t3

    if tmp2 < 0:
        print("--------------------------")
        print(tmp2)
        print(tmp)
        print(random_features)
        print("--------------------------")
    return [tmp, tmp2, tmp3, len(random_features), t1, t2, t3]


def sample_indices_and_subsets(number: int, file: str):
    wine_data = pd.read_csv(file)

    all_features = list(wine_data.columns.values)[:-1]
    featue_number = len(all_features)
    samples = []
    for i in range(1, featue_number + 1):
        for _ in range(number // featue_number):
            random_features = random.sample(all_features, i)
            random_point = random.sample(range(0, len(wine_data)), 1)[0]
            samples.append((random_features, random_point))

    """if number == 1:
        random_features = ["chlorides"]  # random.sample(all_features, 1)
        random_point = 252#random.sample(range(0, len(wine_data)), 1)[0]
        samples.append((random_features, random_point))
    
    print(samples)
    for i in range(0, featue_number):
        print(wine_data.iloc[[random_point], i])"""
    """
    samples = [(['longitude', 'latitude', 'households', 
               'housing_median_age', 'median_income', 
               'total_rooms', 'total_bedrooms'], 3481)]
    """
    # samples = [(['total_bedrooms', 'longitude'], 2856)]
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
    with open(f"{result_path}/samples.pkl", "wb") as fp:  # Pickling
        pickle.dump(samples, fp)

    for s in stddevs:
        for i in iterations:
            run_experiment(i, s, result_path, model_path, data_path, proc_num, samples)
