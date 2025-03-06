import argparse

from cherryml._cherryml_public_api import cherryml_public_api
from cherryml.markov_chain import get_lg_path


def none_or_value(value):
    if value == "None":
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CherryML applied to the LG and co-evolution models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Filepath where to write the learned rate matrix",  # noqa
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help='Either "LG" or "co-evolution". If "LG", a 20x20 rate matrix will be learned. If "co-evolution", a 400x400 rate matrix will be learned.',  # noqa
    )
    parser.add_argument(
        "--msa_dir",
        type=str,
        required=True,
        help="Directory where the training multiple sequence alignments (MSAs) are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files.",  # noqa
    )
    parser.add_argument(
        "--contact_map_dir",
        type=none_or_value,
        required=False,
        default=None,
        help="Directory where the contact maps are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files.",  # noqa
    )
    parser.add_argument(
        "--tree_dir",
        type=none_or_value,
        required=False,
        default=None,
        help="Directory where the trees are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files. If not provided, trees will be estimated with the provided `tree_estimator_name`.",  # noqa
    )
    parser.add_argument(
        "--site_rates_dir",
        type=none_or_value,
        required=False,
        default=None,
        help="Directory where the site rates are stored. See README at https://github.com/songlab-cal/CherryML for the expected format of these files. If not provided, site rates will be estimated with the provided `tree_estimator_name`.",  # noqa
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
        default=32,
        help="Number of processes to parallelize tree estimation (with FastTree).",  # noqa
    )
    parser.add_argument(
        "--num_processes_counting",
        type=int,
        required=False,
        default=1,
        help="Number of processes to parallelize counting transitions.",  # noqa
    )
    parser.add_argument(
        "--num_processes_optimization",
        type=int,
        required=False,
        default=1,
        help="Number of processes to parallelize optimization (if using cpu).",  # noqa
    )

    parser.add_argument(
        "--num_rate_categories",
        type=int,
        required=False,
        default=20,
        help="Number of rate categories to use in FastTree to estimate trees and site rates (if trees are not provided).",  # noqa
    )
    parser.add_argument(
        "--initial_tree_estimator_rate_matrix_path",
        type=str,
        required=False,
        default=get_lg_path(),
        help="Rate matrix to use in FastTree to estimate trees and site rates (the first time around, and only if trees and site rates are not provided)",  # noqa
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        required=False,
        default=1,
        help="Number of times to iterate tree estimation and rate matrix estimation. For highly accurate rate matrix estimation this is a good idea, although tree reconstruction becomes the bottleneck.",  # noqa
    )
    parser.add_argument(
        "--quantization_grid_center",
        type=float,
        required=False,
        default=0.03,
        help="The center value used for time quantization.",  # noqa
    )
    parser.add_argument(
        "--quantization_grid_step",
        type=float,
        required=False,
        default=1.1,
        help="The geometric spacing between time quantization points.",  # noqa
    )
    parser.add_argument(
        "--quantization_grid_num_steps",
        type=int,
        required=False,
        default=64,
        help="The number of quantization points to the left and right of the center.",  # noqa
    )
    parser.add_argument(
        "--use_cpp_counting_implementation",
        type=str,
        required=False,
        default="True",
        help="Whether to use C++ MPI implementation to count transitions ('True' or 'False'). This requires mpirun to be installed. If you do not have mpirun installed, set this argument to False to use a Python implementation (but it will be much slower).",  # noqa
    )
    parser.add_argument(
        "--optimizer_device",
        type=str,
        required=False,
        default="cpu",
        help='Either "cpu" or "cuda". Device to use in PyTorch. "cpu" is fast enough for applications, but if you have a GPU using "cuda" might provide faster runtime.',  # noqa
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=1e-1,
        help="The learning rate in the PyTorch optimizer.",  # noqa
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=500,
        help="The number of epochs of the PyTorch optimizer.",  # noqa
    )
    parser.add_argument(
        "--minimum_distance_for_nontrivial_contact",
        type=int,
        required=False,
        default=7,
        help="Minimum distance in primary structure used to determine if two site are in non-trivial contact.",  # noqa
    )
    parser.add_argument(
        "--families",
        type=none_or_value,
        nargs="*",
        required=False,
        default=None,
        help="Subset of families on which to run rate matrix estimation. If not provided, all families in the `msa_dir` will be used.",  # noqa
    )
    parser.add_argument(
        "--sites_subset_dir",
        type=none_or_value,
        required=False,
        default=None,
        help="Directory where the subset of sites from each family used to learn the rate matrix are specified. Currently only implemented for the LG model. This enables learning e.g. domain-specific or structure-specific rate matrices. See README at https://github.com/songlab-cal/CherryML for the expected format of these files.",  # noqa
    )
    parser.add_argument(
        "--tree_estimator_name",
        type=str,
        required=False,
        default="FastTree",
        help="Tree estimator to use. Can be either 'FastTree' or 'PhyML' or 'FastCherries'. ('FastCherries' is incredibly fast!)",  # noqa
    )
    parser.add_argument(
        "--cherryml_type",
        type=str,
        required=False,
        default="cherry++",
        help="Whether to use 'cherry' or 'cherry++'. Here, 'cherry' uses just "
        "the cherries in the trees, whereas 'cherry++' iteratively picks "
        "cherries until at most one unpaired sequence remains. Thus, 'cherry++'"
        " uses more of the available data. Empirically, 'cherry++' shows "
        "increased statistical efficiency at essentially no runtime cost.",
    )

    # Functionality not currently exposed:
    # parser.add_argument("--do_adam")
    # parser.add_argument("--cpp_counting_command_line_prefix")
    # parser.add_argument("--cpp_counting_command_line_suffix")
    # parser.add_argument("--optimizer_initialization")
    # parser.add_argument("--coevolution_mask_path")
    # parser.add_argument("--use_maximal_matching")

    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict["use_cpp_counting_implementation"] not in ["True", "False"]:
        raise ValueError(
            'use_cpp_counting_implementation should be either "True" or '
            '"False". You provided: '
            f'{args_dict["use_cpp_counting_implementation"]}'
        )
    args_dict["use_cpp_counting_implementation"] = (
        True
        if args_dict["use_cpp_counting_implementation"] == "True"
        else False
    )
    cherryml_public_api(**args_dict)
