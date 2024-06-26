from src.scripts.python_tree_scripts.run_experiment import main

if __name__ == "__main__":
    experiment_type = "ranking"
    stddev_list = "0.01,0.03,0.1,0.3"
    iterations = "100"
    result_path = "results/housing_model/ranking/"
    data_path = "data/housing_data/test_housing_scaled.csv"
    model_path = "models/housing_saved.json"
    proc_num = "30"
    ranking_method = "shap"
    data_name = "housing"

    experiment_args = [
        "--experiment-type",
        experiment_type,
        "--stddev-list",
        stddev_list,
        "--iterations",
        iterations,
        "--results-dir",
        result_path,
        "--data",
        data_path,
        "--model",
        model_path,
        "--proc-num",
        proc_num,
        "--ranking-method",
        ranking_method,
        "--data-name",
        data_name
    ]

    main(experiment_args)
