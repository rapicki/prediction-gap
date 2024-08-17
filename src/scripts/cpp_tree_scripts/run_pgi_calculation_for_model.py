from src.scripts.cpp_tree_scripts.run_experiment import main

if __name__ == "__main__":
    experiment_type = "pgi"
    stddev = "0.1"
    result_path = "results/wine_model/pgi/"
    data_path = "data/wine_quality/test_winequality_red_scaled.csv"
    model_path = "models/winequality_red_saved.json"
    proc_num = "1"
    features_number = "11"
    ranking_file = (
        "results/wine_model_single/ranking/wine_exact_ranking_std_0.3.npy"
    )

    experiment_args = [
        "--experiment-type",
        experiment_type,
        "--stddev",
        stddev,
        "--results-dir",
        result_path,
        "--data",
        data_path,
        "--model",
        model_path,
        "--proc-num",
        proc_num,
        "--features-number",
        features_number,
        "--ranking-file",
        ranking_file,
    ]

    main(experiment_args)
