from src.scripts.cpp_tree_scripts.run_experiment import main

if __name__ == "__main__":
    experiment_type = "ranking"
    stddev_list = "0.3"
    iterations = "1"
    result_path = "results/telemetry_model/ranking/"
    data_path = "data/telemetry/test_telemetry_scaled.csv"
    model_path = "models/telemetry_saved.json"
    proc_num = "3"
    ranking_method = "exact"
    data_name = "telemetry"

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
        data_name,
    ]

    main(experiment_args)
