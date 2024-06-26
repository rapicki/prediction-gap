import argparse

from src.scripts.cpp_tree_scripts.compare_times import compare_times_main


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-type",
        type=str,
        help="Type of experiment to do, has to be one of: pgi, ranking, compare",
    )

    parser.add_argument(
        "--stddev", type=float, default=0.3, help="Standart deviaton used in PGI"
    )
    parser.add_argument(
        "--stddev-list", type=str, help="Standart deviatons [list] used in experiment"
    )
    parser.add_argument("--proc-num", type=int, default=4, help="Number of processors")
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument("--topk", type=int, help="Top k features to take into ranking")
    parser.add_argument(
        "--ranking-dir", type=str, help="Directory with precalculated rankings"
    )
    parser.add_argument("--results-dir", type=str, help="Directory to save results")
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations to perform in case of approx method",
    )
    parser.add_argument(
        "--iterations-list",
        type=str,
        help="Iterations [list] used to perform in case of approx method",
    )
    parser.add_argument(
        "--ranking-method", type=str, help="Method for calculating ranking"
    )
    parser.add_argument(
        "--samples", type=int, help="Naumber of samples to average over"
    )
    parser.add_argument("--features-number", type=int, help="Number of features")
    parser.add_argument("--ranking-file", type=str, help="File with ranking")
    parser.add_argument("--data-name", type=str, help="File with ranking")

    return parser


def parse_args(args=None):
    parser = get_parser()
    return parser.parse_args(args=args)


# if __name__ == "__main__":
def main(args=None):
    args = parse_args(args)
    experiment_type = args.experiment_type
    if experiment_type == "compare":
        compare_times_main(args)
