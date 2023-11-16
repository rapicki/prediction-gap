import os
while "notebooks" in os.getcwd():
    os.chdir("../")
from pathlib import Path
import pandas as pd
import numpy as np
import json

from multiprocessing import Pool

from src.decision_tree.tree import load_trees
from src.decision_tree.prediction_gap import (
    NormalPredictionGap,
    prediction_gap_on_single_feature_perturbation,
    prediction_gap_by_random_sampling,
    prediction_gap_by_exact_calc,
    rank_features_by_random,
)

def calculate_rankings(stddev: float, proc_number: int, num_iter: int):
    
    models_path = Path("models")
    data_path = Path("data")
    results_path = Path("results")
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)
    predgap = NormalPredictionGap(stddev)
    points = []

    for i in range(0, 2):#len(wine_data)):
        points.append((wine_trees, wine_data.iloc[i, :-1]))    
    pool = Pool(processes=proc_num)
    results = np.array(pool.starmap(predgap.rank_features, points))
    np.save(results_path / ("wine" + "_exact_ranking.npy"), results)
 
    points = []
    for i in range(0, 2):# len(wine_data)):
        points.append((wine_trees, wine_data.iloc[i, :-1], stddev, num_iter))    
    pool = Pool(processes=proc_num)
    results = np.array(pool.starmap(rank_features_by_random, points))
    np.save(results_path / ("wine" + f"_approx_ranking_{num_iter}_iter.npy"), results)




if __name__ == "__main__":
    stdev = 0.3
    proc_num = 2
    num_iter = 100
    calculate_rankings(stdev, proc_num, num_iter)