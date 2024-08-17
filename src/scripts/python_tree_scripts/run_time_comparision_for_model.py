from src.scripts.python_tree_scripts.run_experiment import main

if __name__ == "__main__":
    experiment_type = "compare"
    stddev_list = "0.01"  # ,0.03,0.1,0.3"
    iterations = "100"#,500"  # ,1000,1500,2000,4000,5000,6000,10000"#,3000,4000,5000,6000"#,7000,8000,9000,10000,15000"
    result_path = "results/housing_model/test_tiny_times/"
    data_path = "data/housing_data/test_housing_scaled.csv"
    model_path = "models/housing_saved.json"
    proc_num = "1"
    samples = "100"

    experiment_args = [
        "--experiment-type",
        experiment_type,
        "--stddev-list",
        stddev_list,
        "--iterations-list",
        iterations,
        "--results-dir",
        result_path,
        "--data",
        data_path,
        "--model",
        model_path,
        "--proc-num",
        proc_num,
        "--samples",
        samples,
    ]

    main(experiment_args)
