from tree_package import *
import numpy as np
from pathlib import Path
import json
import pandas as pd


class TreeWrapper:
    def __init__(self, model_name: str, model_path: Path) -> None:
        b = self.get_bias_from_file(model_name, model_path)
        p = str(model_path.resolve()) + f"/{model_name}_dumped.txt"
        self.t = TreeParser(p, b)

    @staticmethod
    def get_bias_from_file(model_name: str, model_path: Path):
        with open(model_path / f"{model_name}_saved.json", "r") as fd:
            model = json.load(fd)
        return np.float32(model["learner"]["learner_model_param"]["base_score"])

    def eval(self, df: pd.Series):
        # print(df)
        array = df.to_numpy()
        names = df.keys().values
        s = self.t.eval(array, names)
        return s

    def eval_df(self, df: pd.DataFrame):
        results = []
        for i in range(len(df)):
            point = df.iloc[i, :]
            results.append(self.eval(point))
        return results
