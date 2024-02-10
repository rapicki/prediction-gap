from pathlib import Path
import pandas as pd
import numpy as np

model_name = "wine_model"
results_path = "results"

def get_features(data_name: str) -> list:
    if data_name == "wine":
        fp = Path("data/wine_quality/test_winequality_red_scaled.csv")
    if data_name == "housing":
        fp = Path("data/housing_data/test_housing_scaled.csv")
    data = pd.read_csv(fp)
    return list(data.columns[:-1])

def calculate_entropy(v: np.array) -> float:
    v /= v.sum()
    surprisals = [0.0 if p == 0.0 else p * np.log2(p) for p in v]
    return -sum(surprisals)

def get_rewards(entropy_mode: str, n: int) -> list:
    if entropy_mode == "geometric":
        return [2**(-i) for i in range(n)]
    if entropy_mode.split("_")[0] == "top":
        k = int(entropy_mode.split("_")[1])
        return [1] * k + [0] * (n - k)

def get_rewards_dict(model_name: str, ranking_name: str, entropy_mode: str) -> float:
    features = get_features(model_name.split("_")[0])
    rewards = get_rewards(entropy_mode, len(features))
    ranking_path = Path(results_path) / model_name / "ranking" / f"{ranking_name}.npy"
    rankings = np.load(ranking_path)
    rewards_dict = {feature: 0.0 for feature in features}
    for ranking in rankings:
        for i in range(len(features)):
            rewards_dict[ranking[i]] += rewards[i]
    return rewards_dict

def main():
    name = model_name.split("_")[0]
    stddevs = ["0.01", "0.03", "0.1", "0.3"]
    shap_name = f"{name}_shap_ranking"
    exact_names = [f"{name}_exact_ranking_std_{s}" for s in stddevs]
    approx_names = [f"{name}_approx_ranking_{s}_stddev"
                    for s in ["100_iter_0.01", "100_iter_0.03", "500_iter_0.1", "1500_iter_0.3"]]
    all_names = [shap_name] + exact_names + approx_names
    entropy_results = {ranking_name: [] for ranking_name in all_names}
    entropy_modes = ["geometric", "top_1", "top_2", "top_3"]
    for mode in entropy_modes:
        for ranking_name in all_names:
            rewards_dict = get_rewards_dict(model_name, ranking_name, mode)
            entropy = calculate_entropy(np.array(list(rewards_dict.values())))
            entropy_results[ranking_name].append(entropy)
    df = pd.DataFrame.from_dict(entropy_results, orient="index", columns=entropy_modes)
    df.to_csv(Path(results_path) / model_name / f"{name}_entropy.csv")

if __name__ == "__main__":
    main()
    
