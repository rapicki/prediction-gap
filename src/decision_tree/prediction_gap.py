from scipy.stats import norm
import decision_tree.tree as tree


class PerturbPredictionGap:
    def _compute_cdf(self, data_point, perturbed_features: set):
        raise NotImplementedError('')

    def prediction_gap_fixed(self, model: tree.Model, data_point, baseline_pred, perturbed_features: set):
        return model.expected_diff_squared(self._compute_cdf(data_point, perturbed_features), baseline_pred)

    def pgi(self, model: tree.Model, data_point, baseline_pred, sorted_features: list):
        n = len(sorted_features)
        result = 0
        for k in range(1, n + 1):
            result += self.prediction_gap_fixed(model, data_point, baseline_pred, set(sorted_features[0:k]))
        result /= n

    def pgu(self, model: tree.Model, data_point, baseline_pred, sorted_features: list):
        return self.pgi(model, data_point, baseline_pred, sorted_features[::-1])

    def prediction_gap_single_f(self, model: tree.Model, data_point, perturbed_feature, f):
        return model.expected_single_feature(data_point, perturbed_feature,
                                             self._compute_cdf(data_point, {perturbed_feature})[perturbed_feature],
                                             f)

    def prediction_gap_single_abs(self, model: tree.Model, data_point, perturbed_feature, baseline):
        return self.prediction_gap_single_f(model, data_point, perturbed_feature, lambda x: abs(x - baseline))

    def prediction_gap_single_squared(self, model: tree.Model, data_point, perturbed_feature, baseline):
        return self.prediction_gap_single_f(model, data_point, perturbed_feature, lambda x: (x - baseline) ** 2)


class NormalPredictionGap(PerturbPredictionGap):
    def __init__(self, stddev):
        self.stddev = stddev

    def _compute_cdf(self, data_point, perturbed_features: set):
        cdf_dict = dict()
        for feature, value in data_point.items():
            if feature in perturbed_features:
                cdf_dict[feature] = lambda x: norm.cdf(x, loc=value, scale=self.stddev)
            else:
                # co python robi z przechwytywaniem bez tego t jest powalone
                cdf_dict[feature] = lambda x, t = value: 0.0 if x < t else 1.0
        return cdf_dict
