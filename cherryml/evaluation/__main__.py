import argparse

from cherryml.evaluation._evaluation_public_api import evaluation_public_api


def none_or_value(value):
    if value == "None":
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log-likelihood evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Filepath where to write the log-likelihood",  # noqa
    )
    parser.add_argument(
        "--rate_matrix_path",
        type=str,
        required=True,
        help="Filepath where the rate matrix to evaluate is stored.",  # noqa
    )
    parser.add_argument(
        "--msa_dir",
        type=str,
        required=True,
        help="Directory where the multiple sequence alignments (MSAs) are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files.",  # noqa
    )
    parser.add_argument(
        "--cache_dir",
        type=none_or_value,
        required=False,
        default=None,
        help="Directory to use to cache intermediate computations for re-use in future runs of cherryml. Use a different cache directory for different input datasets. If not provided, a temporary directory will be used.",  # noqa
    )

    parser.add_argument(
        "--num_processes_tree_estimation",
        type=int,
        required=False,
        default=4,
        help="Number of processes to parallelize tree estimation.",  # noqa
    )

    parser.add_argument(
        "--num_rate_categories",
        type=int,
        required=False,
        default=20,
        help="Number of rate categories to use in the tree estimator to estimate trees and site rates.",  # noqa
    )

    parser.add_argument(
        "--families",
        type=none_or_value,
        nargs="*",
        required=False,
        default=None,
        help="Subset of families for which to evaluate log likelihood. If not provided, all families in the `msa_dir` will be used.",  # noqa
    )
    parser.add_argument(
        "--tree_estimator_name",
        type=str,
        required=False,
        default="FastTree",
        help="Tree estimator to use. Can be either 'FastTree' or 'PhyML'.",  # noqa
    )

    parser.add_argument(
        "--extra_command_line_args",
        type=none_or_value,
        required=False,
        default=None,
        help="Extra command line arguments for the tree estimator, e.g. `-gamma` for FastTree to compute Gamma likelihoods.",  # noqa
    )

    args = parser.parse_args()
    args_dict = vars(args)
    evaluation_public_api(**args_dict)
