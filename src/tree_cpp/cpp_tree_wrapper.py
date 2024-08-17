from numpy.core.fromnumeric import size
from tree_package import *
from scipy.stats import qmc, norm
from typing import Optional
import numpy as np
from pathlib import Path
import json
import pandas as pd


class TreeWrapper:
    def __init__(self, model_name: str, model_path: Path) -> None:
        b = self.get_bias_from_file(model_name, model_path)
        p = str(model_path.resolve()) + f"/{model_name}_dumped.txt"
        self.bias = b
        self.t = TreeParser(p, b)

    @staticmethod
    def get_bias_from_file(model_name: str, model_path: Path):
        with open(model_path / f"{model_name}_saved.json", "r") as fd:
            model = json.load(fd)
        return np.float32(model["learner"]["learner_model_param"]["base_score"])

    def eval(self, df: pd.Series):
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

    def prediction_gap_sampling_fast(
        self,
        data_point: pd.Series,
        perturbed_features: set,
        stddev: float = 1.0,
        seed: Optional[int] = None,
        num_iter: int = 100,
    ):
        indx = []
        for i, name in enumerate(data_point.keys().values):
            if name in perturbed_features:
                indx.append(i)

        rng = np.random.default_rng(seed=seed)
        perturbed = np.repeat(
            np.expand_dims(np.array(data_point), axis=0), repeats=num_iter, axis=0
        )
        perturbed[:, indx] = perturbed[:, indx] + rng.normal(
            loc=0.0, scale=stddev, size=(num_iter, len(perturbed_features))
        )

        array = data_point.to_numpy()
        names = data_point.keys().values
        s = self.t.eval_fast(perturbed, array, names)
        return s

    def prediction_gap_sampling_fast_quasi(
        self,
        data_point: pd.Series,
        perturbed_features: set,
        stddev: float = 1.0,
        seed: Optional[int] = None,
        num_iter: int = 100,
    ):
        indx = []
        for i, name in enumerate(data_point.keys().values):
            if name in perturbed_features:
                indx.append(i)

        qrng = qmc.Halton(d=len(perturbed_features))
        sample_qmc = qrng.random(n=num_iter)
        sample_norm = norm.ppf(sample_qmc)
        perturbed = np.repeat(
            np.expand_dims(np.array(data_point), axis=0), repeats=num_iter, axis=0
        )
        perturbed[:, indx] = perturbed[:, indx] + stddev * sample_norm

        array = data_point.to_numpy()
        names = data_point.keys().values
        s = self.t.eval_fast(perturbed, array, names)
        return s

    def prediction_gap_sampling(
        self,
        data_point: pd.Series,
        perturbed_features: set,
        stddev: float = 1.0,
        seed: Optional[int] = None,
        num_iter: int = 100,
    ):
        def normal_perturbation(dp: pd.Series):
            perturbed_dp = dp.copy()
            perturbed_dp[list(perturbed_features)] += rng.normal(
                loc=0.0, scale=stddev, size=len(perturbed_features)
            )
            return perturbed_dp

        y = self.eval(data_point)
        rng = np.random.default_rng(seed=seed)
        result = 0.0
        for i in range(num_iter):
            perturbed_y = self.eval(normal_perturbation(data_point))
            result += (perturbed_y - y) ** 2
        result /= num_iter
        return result

    def prediction_gap_exact(
        self, data_point: pd.Series, perturbed_features: set, stddev: float = 1.0
    ):
        array = data_point.to_numpy()
        names = list(data_point.keys().values)
        # print(perturbed_features)
        s = self.t.expected_diff_squared(
            array,
            names,
            list(perturbed_features),
            stddev,
        )
        return s

    def rank_features(self, data_point: pd.Series, stddev: float = 1.0):
        ranked_features = []
        array = data_point.to_numpy()
        names = list(data_point.keys().values)
        curr_features = set(names)

        while len(curr_features) > 1:
            predgap_dict = {}
            for feature in curr_features:
                tmp = self.t.expected_diff_squared(
                    array,
                    names,
                    list(set(ranked_features) | {feature}),
                    stddev,
                )
                predgap_dict[feature] = tmp
                # print(feature, tmp)
            best_feature = max(
                predgap_dict, key=predgap_dict.get
            )  # this is the current feature with max predgap
            ranked_features.append(best_feature)
            # print(f"Rank {len(ranked_features)}: {best_feature}.")
            curr_features -= {best_feature}
        ranked_features.append(
            list(curr_features)[0]
        )  # this appends the last (and thus lowest ranked) feature
        # print(ranked_features)
        return ranked_features

    def rank_features_by_random(
        self, data_point: pd.Series, stddev: float, num_iter: int
    ):
        ranked_features = []
        names = list(data_point.keys().values)
        curr_features = set(names)

        while len(curr_features) > 1:
            predgap_dict = {}

            for feature in curr_features:
                tmp = self.prediction_gap_sampling_fast_quasi(
                    data_point,
                    list(set(ranked_features) | {feature}),
                    stddev,
                    seed=None,
                    num_iter=num_iter,
                )
                predgap_dict[feature] = tmp
                # print(feature, tmp)
            best_feature = max(
                predgap_dict, key=predgap_dict.get
            )  # this is the current feature with max predgap
            ranked_features.append(best_feature)
            # print(f"Rank {len(ranked_features)}: {best_feature}.")
            curr_features -= {best_feature}
        ranked_features.append(
            list(curr_features)[0]
        )  # this appends the last (and thus lowest ranked) feature
        # print(ranked_features)
        return ranked_features
