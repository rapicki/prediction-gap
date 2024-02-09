from pathlib import Path
import pandas as pd
import numpy as np

model_name = "housing_model_single"
results_path = "new_results_mr"

def get_features(data_name: str) -> list:
    if data_name == "wine":
        fp = Path("data/wine_quality/test_winequality_red_scaled.csv")
    if data_name == "housing":
        fp = Path("data/housing_data/test_housing_scaled.csv")
    data = pd.read_csv(fp)
    return list(data.columns[:-1])

def calculate_entropy(v: np.array) -> float:
    v /= v.sum()
    return -np.dot(v, np.log2(v))

def get_rewards_dict(model_name: str, ranking_name: str) -> float:
    features = get_features(model_name.split("_")[0])
    rewards = [2**(-i) for i in range(len(features))] # that's kinda dumb
    ranking_path = Path(results_path) / model_name / "ranking" / f"{ranking_name}.npy"
    rankings = np.load(ranking_path)
    rewards_dict = {feature: 0.0 for feature in features}
    for ranking in rankings:
        for i in range(len(features)):
            rewards_dict[ranking[i]] += rewards[i]
    return rewards_dict

def main():
    name = model_name.split("_")[0]
    stddevs = ["0.01", "0.03", "0.1", "0.3", "1.0"]
    shap_name = f"{name}_shap_ranking"
    exact_names = [f"{name}_exact_ranking_std_{s}" for s in stddevs]
    approx_names = [f"{name}_approx_ranking_100_iter_{s}_stddev" for s in stddevs]
    all_names = [shap_name] + exact_names + approx_names
    entropy_results = {}
    for ranking_name in all_names:
        rewards_dict = get_rewards_dict(model_name, ranking_name)
        entropy = calculate_entropy(np.array(list(rewards_dict.values())))
        entropy_results[ranking_name] = entropy
    df = pd.DataFrame.from_dict({k: [v] for k, v in entropy_results.items()},
                                orient="index", columns=["entropy"])
    df.to_csv(Path(results_path) / model_name / f"{name}_entropy.csv")

if __name__ == "__main__":
    main()
    
