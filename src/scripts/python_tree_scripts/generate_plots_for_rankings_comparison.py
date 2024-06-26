from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def create_plot(results_path: Path, model_name: str, pgi_name: str, k: str):
    """Example args:

    results_path=Path("new_results_mr"),
    model_name="wine_model_single",
    pgi_name="pgi_std_0.1",
    k="mean"
    """
    name = model_name.split("_")[0]
    stddevs = ["0.01", "0.03", "0.1", "0.3", "1.0"]
    # if "single" in model_name:
    #    stddevs.append("1.0")
    exact_ranking_names = [f"{name}_exact_ranking_std_{s}" for s in stddevs]
    ranking_names = exact_ranking_names + [f"{name}_shap_ranking"]
    exact_results = {
        ranking_name: np.mean(
            np.load(
                results_path / model_name / "pgi" / ranking_name / f"{pgi_name}_{k}.npy"
            )
        )
        for ranking_name in exact_ranking_names
    }
    shap_result = np.mean(
        np.load(
            results_path
            / model_name
            / "pgi"
            / f"{name}_shap_ranking"
            / f"{pgi_name}_{k}.npy"
        )
    )

    plt.figure()
    plt.plot(stddevs, exact_results.values(), label="PG rankings")
    plt.plot(
        stddevs, [shap_result] * len(stddevs), linestyle="dashed", label="SHAP rankings"
    )
    plt.xlabel("Standard deviation used for PG feature ranking")
    plt.ylabel("Average PGI² of the rankings")
    plt.title(f"Rankings comparison by PGI² with σ={pgi_name.split('_')[-1]}")
    plt.legend(frameon=True)
    plt.savefig(Path("plots") / f"{model_name}_{pgi_name}_{k}.jpg")
    plt.close()


if __name__ == "__main__":
    # plt.style.use('science')

    # medium models
    results_path = Path("results")
    for s in ["0.01", "0.03", "0.1", "0.3", "1.0"]:
        create_plot(results_path, "wine_model", f"pgi_std_{s}", "mean")
        create_plot(results_path, "housing_model", f"pgi_std_{s}", "mean") # no data yet

    # single models
    results_path = Path("results")
    for s in ["0.01", "0.03", "0.1", "0.3", "1.0"]:
        create_plot(results_path, "wine_model_single", f"pgi_std_{s}", "mean")
        create_plot(results_path, "housing_model_single", f"pgi_std_{s}", "mean")
