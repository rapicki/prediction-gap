import os

from pandas.core.algorithms import rank
from pandas.core.dtypes.dtypes import re
while "notebooks" in os.getcwd():
    os.chdir("../")
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from multiprocessing import Pool

from src.decision_tree.tree import load_trees
from src.decision_tree.prediction_gap import PerturbPredictionGap
import random
    
def calculate_rankings(stddev: float, proc_number: int, 
                       ranking_names: list[str], pgi_names: list[str]):
    

    models_path = Path("models")
    data_path = Path("data")
    results_path = Path("results")
    
    wine_model_name = "winequality_red"
    wine_test_data_path = data_path / "wine_quality/test_winequality_red_scaled.csv"
    wine_trees = load_trees(models_path, wine_model_name)
    wine_data = pd.read_csv(wine_test_data_path)
    
    predgap = PerturbPredictionGap(stddev)
    wine_trees = load_trees(models_path, wine_model_name)
    
    for rnk, pgi in zip(ranking_names, pgi_names):
        ranking = np.load(results_path/rnk)
        points = [] 
        for i in range(0, 3):#len(wine_data)):
            data_point = wine_data.iloc[0, :]
            baseline_pred = wine_trees.eval(data_point)

            points.append((wine_trees, wine_data.iloc[i, :-1], baseline_pred, 
                          
                           ranking[i].tolist()))    
        pool = Pool(processes=proc_number)
        results = np.array(pool.starmap(predgap.pgi, points))
        np.save((results_path / pgi), results)

if __name__ == "__main__":
    rankings_names = ["wine_exact_ranking.npy", "wine_shap_ranking.npy",
            "wine_approx_ranking_4000_iter.npy"]
    pgi_file_names = ["pgi_wine_exact_ranking.npy", "pgi_wine_shap_ranking.npy",
            "pgi_wine_approx_ranking_4000_iter.npy"
            ]
    stdev = 0.3
    proc_num = 20
    calculate_rankings(stdev, proc_num, rankings_names, pgi_file_names)
