"""
Module to reproduce and extend all figures.

trRosetta dataset: https://www.pnas.org/doi/10.1073/pnas.1914677117

Prerequisites:
- input_data/a3m should point to the trRosetta alignments (e.g. via a symbolic
    link)
- input_data/pdb should point to the trRosetta structures (e.g. via a symbolic
    link)

The caching directories which contain all subsequent data are
_cache_benchmarking and _cache_benchmarking_em. You can similarly use a symbolic
link to point to these.
"""
import logging
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

import cherryml.utils as utils
from cherryml import (
    caching,
    coevolution_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_em_optimizer,
)
from cherryml.benchmarking.lg_paper import (
    get_lg_PfamTestingAlignments_data,
    get_lg_PfamTrainingAlignments_data,
    reproduce_lg_paper_fig_4,
)
from cherryml.benchmarking.pfam_15k import compute_contact_maps
from cherryml.benchmarking.pfam_15k import get_families as get_families_pfam_15k
from cherryml.benchmarking.pfam_15k import (
    get_families_within_cutoff,
    simulate_ground_truth_data_coevolution,
    simulate_ground_truth_data_single_site,
    subsample_pfam_15k_msas,
)
from cherryml.evaluation import (
    compute_log_likelihoods,
    create_maximal_matching_contact_map,
    plot_rate_matrix_predictions,
    relative_errors,
)
from cherryml.global_vars import TITLES
from cherryml.io import (
    read_contact_map,
    read_log_likelihood,
    read_mask_matrix,
    read_msa,
    read_rate_matrix,
    read_site_rates,
    write_count_matrices,
    write_probability_distribution,
    write_rate_matrix,
    write_sites_subset,
)
from cherryml.markov_chain import (
    chain_product,
    compute_mutation_rate,
    compute_stationary_distribution,
    get_aa_coevolution_mask_path,
    get_equ_path,
    get_jtt_path,
    get_lg_path,
    get_wag_path,
    matrix_exponential,
    normalized,
)
from cherryml.phylogeny_estimation import fast_tree, gt_tree_estimator, phyml
from cherryml.types import PhylogenyEstimatorType
from cherryml.utils import get_families, get_process_args


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


PFAM_15K_MSA_DIR = "input_data/a3m"
PFAM_15K_PDB_DIR = "input_data/pdb"


def add_annotations_to_violinplot(
    yss_relative_errors: List[float],
    title: str,
    xlabel: str,
    runtimes: Optional[List[float]] = None,
    grid: bool = True,
    fontsize: int = 14,
):
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    yticks = [np.log(10**i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
    if grid:
        plt.grid()
    plt.ylabel("Relative error (%)\nDistribution and median", fontsize=fontsize)
    plt.yticks(yticks, ytickslabels, fontsize=fontsize)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
        plt.annotate(
            label,  # this is the text
            (
                i + 0.05,
                np.log(np.max(ys)) - 1.5,
            ),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="left",
            va="top",
            color="black",
            fontsize=fontsize,
        )
        if runtimes is not None:
            label = "{:.0f}s".format(runtimes[i])
            plt.annotate(
                label,  # this is the text
                (
                    i + 0.05,
                    np.log(np.max(ys)),
                ),  # these are the coordinates to position the label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha="left",
                va="top",
                color="blue",
                fontsize=fontsize,
            )
    if TITLES:
        if runtimes is None:
            plt.title(title + "\n(median error also reported)")
        else:
            plt.title(title + "\n(median error and runtime also reported)")
    plt.tight_layout()


@caching.cached_computation(
    output_dirs=["output_count_matrices_dir"],
)
def create_synthetic_count_matrices(
    quantization_points: List[float],
    samples_per_row: int,
    rate_matrix_path: str,
    output_count_matrices_dir: Optional[str] = None,
):
    """
    Create synthetic count matrices.

    Args:
        quantization_points: The branch lengths.
        samples_per_row: For each branch length, this number of transitions will
            be observed for each state (up to transitions lost from taking
            floor)
        rate_matrix_path: Path to the ground truth rate matrix.
    """
    Q_df = read_rate_matrix(rate_matrix_path)
    Q_numpy = Q_df.to_numpy()
    count_matrices = [
        [
            q,
            pd.DataFrame(
                (
                    samples_per_row
                    * matrix_exponential(
                        exponents=np.array([q]),
                        Q=Q_numpy,
                        fact=None,
                        reversible=False,
                        device="cpu",
                    ).reshape([Q_numpy.shape[0], Q_numpy.shape[1]])
                ).astype(int),
                columns=Q_df.columns,
                index=Q_df.index,
            ),
        ]
        for q in quantization_points
    ]
    write_count_matrices(
        count_matrices, os.path.join(output_count_matrices_dir, "result.txt")
    )


def fig_single_site_cherry(
    num_rate_categories: int = 1,
    num_processes_tree_estimation: int = 4,
    num_sequences: int = 128,
    random_seed: int = 0,
):
    caching.set_cache_dir("_cache_benchmarking_em")
    caching.set_hash_len(64)

    output_image_dir = f"images/fig_single_site_cherry"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_families_train_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    runtimes = []
    yss_relative_errors = []
    Qs = []
    for (i, num_families_train) in enumerate(num_families_train_list):
        msg = f"***** num_families_train = {num_families_train} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            min_num_sites=190,
            max_num_sites=230,
            min_num_sequences=num_sequences,
            max_num_sequences=1000000,
        )
        families_train = families_all[:num_families_train]

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_single_site(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences,
            families=families_all,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes_tree_estimation,
            random_seed=random_seed,
        )

        # Now run the cherryml method.
        lg_end_to_end_with_cherryml_optimizer_res = (
            lg_end_to_end_with_cherryml_optimizer(
                msa_dir=msa_dir,
                families=families_train,
                tree_estimator=partial(
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),
                num_processes_tree_estimation=num_processes_tree_estimation,
                num_processes_optimization=1,
                num_processes_counting=1,
                edge_or_cherry="cherry",
            )
        )

        def get_runtime(
            lg_end_to_end_with_cherryml_optimizer_res: str,
        ) -> float:
            """
            Get the runtime of CherryML.
            """
            res = 0
            for lg_end_to_end_with_cherryml_optimizer_output_dir in [
                "count_matrices_dir_0",
                "jtt_ipw_dir_0",
                "rate_matrix_dir_0",
            ]:
                with open(
                    os.path.join(
                        lg_end_to_end_with_cherryml_optimizer_res[
                            lg_end_to_end_with_cherryml_optimizer_output_dir
                        ],
                        "profiling.txt",
                    ),
                    "r",
                ) as profiling_file:
                    profiling_file_contents = profiling_file.read()
                    print(
                        f"{lg_end_to_end_with_cherryml_optimizer_output_dir} "  # noqa
                        f"profiling_file_contents = {profiling_file_contents}"  # noqa
                    )
                    res += float(profiling_file_contents.split()[2])
            return res

        runtime = get_runtime(lg_end_to_end_with_cherryml_optimizer_res)
        runtimes.append(runtime)

        learned_rate_matrix_path = os.path.join(
            lg_end_to_end_with_cherryml_optimizer_res["rate_matrix_dir_0"],
            "result.txt",
        )
        learned_rate_matrix = read_rate_matrix(
            learned_rate_matrix_path
        ).to_numpy()
        lg = read_rate_matrix(get_lg_path()).to_numpy()
        learned_rate_matrix_path = lg_end_to_end_with_cherryml_optimizer_res[
            "learned_rate_matrix_path"
        ]
        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

        learned_rate_matrix = learned_rate_matrix.to_numpy()
        Qs.append(learned_rate_matrix)

        lg = read_rate_matrix(get_lg_path()).to_numpy()

        yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

    for i in range(len(num_families_train_list)):
        plot_rate_matrix_predictions(
            read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
        )
        if TITLES:
            plt.title(
                "True vs predicted rate matrix entries\nnumber of families = %i"
                % num_families_train_list[i]
            )
        plt.tight_layout()
        plt.savefig(
            f"{output_image_dir}/log_log_plot_{i}",
            dpi=300,
        )
        plt.close()

    df = pd.DataFrame(
        {
            "Number of families": sum(
                [
                    [num_families_train_list[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "Relative error": sum(yss_relative_errors, []),
        }
    )
    df["Log relative error"] = np.log(df["Relative error"])

    sns.violinplot(
        x="Number of families",
        y="Log relative error",
        data=df,
        inner=None,
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as sample size increases",
        xlabel="Number of families",
        runtimes=runtimes,
    )
    plt.savefig(
        f"{output_image_dir}/violin_plot",
        dpi=300,
    )
    plt.close()

    ys_relative_errors = [np.median(ys) for ys in yss_relative_errors]
    return num_families_train_list, ys_relative_errors, runtimes


def fig_single_site_em(
    extra_em_command_line_args: str = "-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",  # noqa
    num_processes: int = 4,
    num_rate_categories: int = 1,
    num_sequences: int = 128,
    random_seed: int = 0,
):
    output_image_dir = (
        "images/fig_single_site_em__"
        + extra_em_command_line_args.replace(" ", "_")
    )
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_benchmarking_em")
    caching.set_hash_len(64)

    num_families_train_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    yss_relative_errors = []
    runtimes = []
    Qs = []
    for (i, num_families_train) in enumerate(num_families_train_list):
        msg = f"***** num_families_train = {num_families_train} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            min_num_sites=190,
            max_num_sites=230,
            min_num_sequences=num_sequences,
            max_num_sequences=1000000,
        )
        families_train = families_all[:num_families_train]

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_single_site(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences,
            families=families_all,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes,
            random_seed=random_seed,
        )

        em_estimator_res = lg_end_to_end_with_em_optimizer(
            msa_dir=msa_dir,
            families=families_train,
            tree_estimator=partial(
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            extra_em_command_line_args=extra_em_command_line_args,
        )

        def get_runtime(profiling_file_path: str):
            with open(profiling_file_path, "r") as profiling_file:
                profiling_file_contents = profiling_file.read()
                print(f"profiling_file_contents = {profiling_file_contents}")
                return float(profiling_file_contents.split()[2])

        runtime = get_runtime(
            os.path.join(em_estimator_res["rate_matrix_dir_0"], "profiling.txt")
        )
        runtimes.append(runtime)

        learned_rate_matrix_path = os.path.join(
            em_estimator_res["rate_matrix_dir_0"], "result.txt"
        )
        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
        learned_rate_matrix = learned_rate_matrix.to_numpy()

        lg = read_rate_matrix(get_lg_path()).to_numpy()
        learned_rate_matrix_path = em_estimator_res["learned_rate_matrix_path"]
        learned_rate_matrix = read_rate_matrix(
            learned_rate_matrix_path
        ).to_numpy()
        Qs.append(learned_rate_matrix)

        lg = read_rate_matrix(get_lg_path()).to_numpy()

        yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

    for i in range(len(num_families_train_list)):
        plot_rate_matrix_predictions(
            read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
        )
        if TITLES:
            plt.title(
                "True vs predicted rate matrix entries\nnumber of families = %i"
                % num_families_train_list[i]
            )
        plt.tight_layout()
        plt.savefig(
            f"{output_image_dir}/log_log_plot_{i}",
            dpi=300,
        )
        plt.close()

    df = pd.DataFrame(
        {
            "Number of families": sum(
                [
                    [num_families_train_list[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "Relative error": sum(yss_relative_errors, []),
        }
    )
    df["Log relative error"] = np.log(df["Relative error"])

    sns.violinplot(
        x="Number of families",
        y="Log relative error",
        data=df,
        inner=None,
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as sample size increases",
        xlabel="Number of families",
        runtimes=runtimes,
    )
    plt.savefig(
        f"{output_image_dir}/violin_plot",
        dpi=300,
    )
    plt.close()

    ys_relative_errors = [np.median(ys) for ys in yss_relative_errors]
    return num_families_train_list, ys_relative_errors, runtimes


# Fig. 1b, 1c
def fig_computational_and_stat_eff_cherry_vs_em():
    fontsize = 14

    output_image_dir = "images/fig_computational_and_stat_eff_cherry_vs_em"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    (
        num_families_cherry,
        cherry_errors_nonpct,
        cherry_times,
    ) = fig_single_site_cherry()
    cherry_times = [int(x) for x in cherry_times]
    cherry_errors = [float("%.1f" % (100 * x)) for x in cherry_errors_nonpct]
    num_families_em, em_errors_nonpct, em_times = fig_single_site_em()
    em_times = [int(x) for x in em_times]
    em_errors = [float("%.1f" % (100 * x)) for x in em_errors_nonpct]
    assert num_families_cherry == num_families_em

    print(f"cherry_errors_nonpct = {cherry_errors_nonpct}")
    print(f"cherry_times = {cherry_times}")
    print(f"em_errors_nonpct = {em_errors_nonpct}")
    print(f"em_times = {em_times}")

    num_families = num_families_cherry
    indices = [i for i in range(len(num_families))]
    plt.figure(dpi=300)
    plt.plot(
        indices, 100 * np.array(cherry_errors_nonpct), "o-", label="CherryML"
    )
    plt.plot(indices, 100 * np.array(em_errors_nonpct), "o-", label="EM")
    plt.ylim((0.5, 200))
    plt.xticks(indices, num_families, fontsize=fontsize)
    plt.yscale("log", base=10)
    plt.yticks(fontsize=fontsize)
    plt.grid()
    plt.legend(loc="upper left", fontsize=fontsize)
    plt.xlabel("Number of families", fontsize=fontsize)
    plt.ylabel("Median relative error (%)", fontsize=fontsize)
    for a, b in zip(indices, em_errors):
        plt.text(a - 0.35, b / 1.5, str(b) + "%", fontsize=fontsize)
    for a, b in zip(indices, cherry_errors):
        plt.text(a - 0.3, 1.2 * b, str(b) + "%", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_image_dir, "errors"))
    plt.close()

    num_families = num_families_cherry
    indices = [i for i in range(len(num_families))]
    plt.figure(dpi=300)
    plt.plot(indices, cherry_times, "o-", label="CherryML")
    plt.plot(indices, em_times, "o-", label="EM")
    plt.ylim((5, 5e5))
    plt.xticks(indices, num_families, fontsize=fontsize)
    plt.yscale("log", base=10)
    plt.yticks(fontsize=fontsize)
    plt.grid()
    plt.legend(loc="upper left", fontsize=fontsize)
    plt.xlabel("Number of families", fontsize=fontsize)
    plt.ylabel("Runtime (s)", fontsize=fontsize)
    for a, b in zip(indices, em_times):
        plt.text(a - 0.35, b * 1.5, str(b) + "s", fontsize=fontsize)
    for a, b in zip(indices, cherry_times):
        plt.text(a - 0.3, b * 1.5, str(b) + "s", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_image_dir, "times"))
    plt.close()


# Fig. 1d
def fig_single_site_quantization_error(
    num_rate_categories: int = 4,
    num_processes_tree_estimation: int = 32,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 2,
    num_families_train: int = 15051,
    num_sequences: int = 1024,
    random_seed: int = 0,
):
    """
    We show that ~100 quantization points (geometric increments of 10%) is
    enough.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_single_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    qs = [
        (0.03, 445.79, 1),
        (0.03, 21.11, 2),
        (0.03, 4.59, 4),
        (0.03, 2.14, 8),
        (0.03, 1.46, 16),
        (0.03, 1.21, 32),
        (0.03, 1.1, 64),
        (0.03, 1.048, 128),
        (0.03, 1.024, 256),
    ]
    q_errors = [(np.sqrt(q[1]) - 1) * 100 for q in qs]
    q_points = [2 * q[2] + 1 for q in qs]
    yss_relative_errors = []
    Qs = []
    for (
        i,
        (
            quantization_grid_center,
            quantization_grid_step,
            quantization_grid_num_steps,
        ),
    ) in enumerate(qs):
        msg = f"***** grid = {(quantization_grid_center, quantization_grid_step, quantization_grid_num_steps)} *****"  # noqa
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_train = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR
        )

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_single_site(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences,
            families=families_train,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes_tree_estimation,
            random_seed=random_seed,
        )

        lg_end_to_end_with_cherryml_optimizer_res = (
            lg_end_to_end_with_cherryml_optimizer(
                msa_dir=msa_dir,
                families=families_train,
                tree_estimator=partial(
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),
                quantization_grid_center=quantization_grid_center,
                quantization_grid_step=quantization_grid_step,
                quantization_grid_num_steps=quantization_grid_num_steps,
                num_processes_tree_estimation=num_processes_tree_estimation,
                num_processes_counting=num_processes_counting,
                num_processes_optimization=num_processes_optimization,
            )
        )

        learned_rate_matrix_path = os.path.join(
            lg_end_to_end_with_cherryml_optimizer_res["rate_matrix_dir_0"],
            "result.txt",
        )

        learned_rate_matrix = read_rate_matrix(
            learned_rate_matrix_path
        ).to_numpy()
        Qs.append(learned_rate_matrix)
        lg = read_rate_matrix(get_lg_path()).to_numpy()
        yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

    for i in range(len(q_points)):
        plot_rate_matrix_predictions(
            read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
        )
        if TITLES:
            plt.title(
                "True vs predicted rate matrix entries\nmax quantization error = "
                "%.1f%% (%i quantization points)" % (q_errors[i], q_points[i])
            )
        plt.tight_layout()
        plt.savefig(
            f"{output_image_dir}/log_log_plot_{i}",
            dpi=300,
        )
        plt.close()

    df = pd.DataFrame(
        {
            "Quantization points": sum(
                [
                    [q_points[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "Relative error": sum(yss_relative_errors, []),
        }
    )
    df["Log relative error"] = np.log(df["Relative error"])

    sns.violinplot(
        x="Quantization points",
        y="Log relative error",
        data=df,
        inner=None,
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as quantization improves",
        xlabel="Quantization points",
    )
    plt.savefig(
        f"{output_image_dir}/violin_plot",
        dpi=300,
    )
    plt.close()


# Fig. 1e
def fig_lg_paper(
    num_rate_categories: int = 4,
    figsize=(6.4, 4.8),
    rate_estimator_names: List[Tuple[str, str]] = [
        ("reproduced WAG", "WAG"),
        ("reproduced LG", "LG"),
        ("Cherry__4", "LG w/CherryML"),
    ],
    baseline_rate_estimator_name: Tuple[str, str] = ("reproduced JTT", "JTT"),
    num_processes: int = 4,
):
    """
    LG paper figure 4, extended with LG w/CherryML.
    """
    output_image_dir = "images/fig_lg_paper/"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_lg_paper")
    caching.set_hash_len(64)

    LG_PFAM_TRAINING_ALIGNMENTS_DIR = (
        "./lg_paper_data/lg_PfamTrainingAlignments"
    )
    LG_PFAM_TESTING_ALIGNMENTS_DIR = "./lg_paper_data/lg_PfamTestingAlignments"

    get_lg_PfamTrainingAlignments_data(LG_PFAM_TRAINING_ALIGNMENTS_DIR)
    get_lg_PfamTestingAlignments_data(LG_PFAM_TESTING_ALIGNMENTS_DIR)

    phyml_partial = partial(
        phyml,
        num_rate_categories=num_rate_categories,
        num_processes=num_processes,
    )

    y, df, bootstraps, Qs = reproduce_lg_paper_fig_4(
        msa_train_dir=LG_PFAM_TRAINING_ALIGNMENTS_DIR,
        families_train=get_families(LG_PFAM_TRAINING_ALIGNMENTS_DIR),
        msa_test_dir=LG_PFAM_TESTING_ALIGNMENTS_DIR,
        families_test=get_families(LG_PFAM_TESTING_ALIGNMENTS_DIR),
        rate_estimator_names=rate_estimator_names[:],
        baseline_rate_estimator_name=baseline_rate_estimator_name,
        evaluation_phylogeny_estimator=phyml_partial,
        num_processes=num_processes,
        pfam_or_treebase="pfam",
        family_name_len=7,
        figsize=figsize,
        num_bootstraps=100,
        output_image_dir=output_image_dir,
        use_colors=True,
    )


@caching.cached_computation(
    output_dirs=["output_probability_distribution_dir"],
)
def get_stationary_distribution(
    rate_matrix_path: str,
    output_probability_distribution_dir: Optional[str] = None,
):
    rate_matrix = read_rate_matrix(rate_matrix_path)
    pi = compute_stationary_distribution(rate_matrix.to_numpy())
    write_probability_distribution(
        pi,
        rate_matrix.index,
        os.path.join(output_probability_distribution_dir, "result.txt"),
    )


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
)
def normalize_rate_matrix(
    rate_matrix_path: str,
    new_rate: float,
    output_rate_matrix_dir: Optional[str] = None,
):
    rate_matrix = read_rate_matrix(rate_matrix_path)
    normalized_rate_matrix = new_rate * normalized(rate_matrix.to_numpy())
    write_rate_matrix(
        normalized_rate_matrix,
        rate_matrix.index,
        os.path.join(output_rate_matrix_dir, "result.txt"),
    )


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
)
def chain_product_cached(
    rate_matrix_1_path: str,
    rate_matrix_2_path: str,
    output_rate_matrix_dir: Optional[str] = None,
):
    rate_matrix_1 = read_rate_matrix(rate_matrix_1_path)
    rate_matrix_2 = read_rate_matrix(rate_matrix_2_path)
    res = chain_product(rate_matrix_1.to_numpy(), rate_matrix_2.to_numpy())
    if list(rate_matrix_1.index) != list(rate_matrix_2.index):
        raise Exception(
            "Double-check that the states are being computed correctly in the "
            "code."
        )
    states = [
        state_1 + state_2
        for state_1 in rate_matrix_1.index
        for state_2 in rate_matrix_2.index
    ]
    write_rate_matrix(
        res, states, os.path.join(output_rate_matrix_dir, "result.txt")
    )


def evaluate_single_site_model_on_held_out_msas(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_processes: int,
    tree_estimator: PhylogenyEstimatorType,
):
    """
    Evaluate a reversible single-site model on held out msas.
    """
    # First estimate the trees
    tree_estimator_output_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_processes=num_processes,
    )
    tree_dir = tree_estimator_output_dirs["output_tree_dir"]
    site_rates_dir = tree_estimator_output_dirs["output_site_rates_dir"]
    output_probability_distribution_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_path,
    )["output_probability_distribution_dir"]
    pi_1_path = os.path.join(output_probability_distribution_dir, "result.txt")
    output_likelihood_dir = compute_log_likelihoods(
        tree_dir=tree_dir,
        msa_dir=msa_dir,
        site_rates_dir=site_rates_dir,
        contact_map_dir=None,
        families=families,
        amino_acids=utils.get_amino_acids(),
        pi_1_path=pi_1_path,
        Q_1_path=rate_matrix_path,
        reversible_1=True,
        device_1="cpu",
        pi_2_path=None,
        Q_2_path=None,
        reversible_2=None,
        device_2=None,
        num_processes=num_processes,
        use_cpp_implementation=False,
        OMP_NUM_THREADS=1,
        OPENBLAS_NUM_THREADS=1,
    )["output_likelihood_dir"]
    lls = []
    for family in families:
        ll = read_log_likelihood(
            os.path.join(output_likelihood_dir, f"{family}.txt")
        )
        lls.append(ll[0])
    return np.sum(lls)


def evaluate_pair_site_model_on_held_out_msas(
    msa_dir: str,
    contact_map_dir: str,
    families: List[str],
    rate_matrix_1_path: str,
    rate_matrix_2_path: str,
    num_processes: int,
    tree_estimator: PhylogenyEstimatorType,
):
    """
    Evaluate a reversible single-site model and coevolution model on held out
    msas.
    """
    # First estimate the trees
    tree_estimator_output_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_1_path,
        num_processes=num_processes,
    )
    tree_dir = tree_estimator_output_dirs["output_tree_dir"]
    site_rates_dir = tree_estimator_output_dirs["output_site_rates_dir"]
    output_probability_distribution_1_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_1_path,
    )["output_probability_distribution_dir"]
    pi_1_path = os.path.join(
        output_probability_distribution_1_dir, "result.txt"
    )
    output_probability_distribution_2_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_2_path,
    )["output_probability_distribution_dir"]
    pi_2_path = os.path.join(
        output_probability_distribution_2_dir, "result.txt"
    )
    output_likelihood_dir = compute_log_likelihoods(
        tree_dir=tree_dir,
        msa_dir=msa_dir,
        site_rates_dir=site_rates_dir,
        contact_map_dir=contact_map_dir,
        families=families,
        amino_acids=utils.get_amino_acids(),
        pi_1_path=pi_1_path,
        Q_1_path=rate_matrix_1_path,
        reversible_1=True,
        device_1="cpu",
        pi_2_path=pi_2_path,
        Q_2_path=rate_matrix_2_path,
        reversible_2=True,
        device_2="cpu",
        num_processes=num_processes,
        use_cpp_implementation=False,
        OMP_NUM_THREADS=1,
        OPENBLAS_NUM_THREADS=1,
    )["output_likelihood_dir"]
    lls = []
    for family in families:
        ll = read_log_likelihood(
            os.path.join(output_likelihood_dir, f"{family}.txt")
        )
        lls.append(ll[0])
    return np.sum(lls)


def _map_func_compute_contacting_sites(args: List) -> None:
    contact_map_dir = args[0]
    minimum_distance_for_nontrivial_contact = args[1]
    families = args[2]
    output_sites_subset_dir = args[3]

    for family in families:
        contact_map_path = os.path.join(contact_map_dir, family + ".txt")
        contact_map = read_contact_map(contact_map_path)

        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if abs(i - j) >= minimum_distance_for_nontrivial_contact and i < j
        ]
        contacting_sites = sorted(list(set(sum(contacting_pairs, ()))))
        # This exception below is not needed because downstream code (counting
        # transitions) has no issue with this border case.
        # if len(contacting_sites) == 0:
        #     raise Exception(
        #         f"Family {family} has no nontrivial contacting sites. "
        #         "This would lead to an empty subset."
        #     )

        write_sites_subset(
            contacting_sites,
            os.path.join(output_sites_subset_dir, family + ".txt"),
        )

        caching.secure_parallel_output(output_sites_subset_dir, family)


@caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_sites_subset_dir"],
)
def _compute_contacting_sites(
    contact_map_dir: str,
    minimum_distance_for_nontrivial_contact: int,
    families: List[str],
    num_processes: int = 1,
    output_sites_subset_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info(
        f"Subsetting sites to contacting sites on {len(families)} families "
        f"using {num_processes} processes. "
        f"output_sites_subset_dir: {output_sites_subset_dir}"
    )

    map_args = [
        [
            contact_map_dir,
            minimum_distance_for_nontrivial_contact,
            get_process_args(process_rank, num_processes, families),
            output_sites_subset_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_compute_contacting_sites, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_compute_contacting_sites, map_args),
                total=len(map_args),
            )
        )

    logger.info("Computing contacting sites done!")


def learn_coevolution_model_on_pfam15k(
    num_rate_categories: int = 1,
    num_sequences: int = 1024,
    num_families_train: int = 15051,
    num_families_test: int = 1,
    num_processes_tree_estimation: int = 32,
    num_processes_counting: int = 8,
    num_processes_optimization_single_site: int = 2,
    num_processes_optimization_coevolution: int = 8,
    angstrom_cutoff: float = 8.0,
    minimum_distance_for_nontrivial_contact: int = 7,
) -> Dict:
    """
    Returns a dictionary with the learned rate matrices.

    Test set can be used to compute held-out log-likelihoods.
    """
    output_image_dir = "images/learn_coevolution_model_on_pfam15k"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    PFAM_15K_MSA_DIR = "input_data/a3m"
    PFAM_15K_PDB_DIR = "input_data/pdb"

    train_test_split_seed = 0

    families_all = get_families_pfam_15k(
        PFAM_15K_MSA_DIR,
    )
    np.random.seed(train_test_split_seed)
    np.random.shuffle(families_all)

    families_train = sorted(families_all[:num_families_train])
    if num_families_test == 0:
        families_test = []
    else:
        families_test = sorted(families_all[-num_families_test:])

    # Subsample the MSAs
    msa_dir_train = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families_train,
        num_processes=num_processes_tree_estimation,
    )["output_msa_dir"]

    # Run the cherry method using FastTree tree estimator
    cherry_path = lg_end_to_end_with_cherryml_optimizer(
        msa_dir=msa_dir_train,
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes_tree_estimation=num_processes_tree_estimation,
        num_processes_optimization=num_processes_optimization_single_site,
        num_processes_counting=num_processes_counting,
    )["learned_rate_matrix_path"]
    cherry = read_rate_matrix(cherry_path).to_numpy()
    del cherry  # unused

    lg = read_rate_matrix(get_lg_path()).to_numpy()
    del lg  # unused

    # Subsample the MSAs
    msa_dir_test = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families_test,
        num_processes=num_processes_tree_estimation,
    )["output_msa_dir"]

    log_likelihoods = []  # type: List[Tuple[str, float]]
    single_site_rate_matrices = [
        ("JTT", get_jtt_path()),
        ("WAG", get_wag_path()),
        ("LG", get_lg_path()),
        ("Cherry", cherry_path),
    ]

    for rate_matrix_name, rate_matrix_path in single_site_rate_matrices:
        mutation_rate = compute_mutation_rate(
            read_rate_matrix(rate_matrix_path)
        )
        print(
            f"***** Evaluating: {rate_matrix_name} at {rate_matrix_path} ({num_rate_categories} rate categories) with global mutation rate {mutation_rate} *****"  # noqa
        )
        ll = evaluate_single_site_model_on_held_out_msas(
            msa_dir=msa_dir_test,
            families=families_test,
            rate_matrix_path=rate_matrix_path,
            num_processes=num_processes_tree_estimation,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=num_rate_categories,
            ),
        )
        print(f"ll for {rate_matrix_name} = {ll}")
        log_likelihoods.append((rate_matrix_name, ll))

    # Run the single-site cherry method *ONLY ON CONTACTING SITES*
    contact_map_dir_train = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families_train,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes_tree_estimation,
    )["output_contact_map_dir"]

    mdnc = minimum_distance_for_nontrivial_contact
    contacting_sites_dir = _compute_contacting_sites(
        contact_map_dir=contact_map_dir_train,
        minimum_distance_for_nontrivial_contact=mdnc,
        families=families_train,
        num_processes=num_processes_tree_estimation,
    )["output_sites_subset_dir"]

    cherry_contact_path = lg_end_to_end_with_cherryml_optimizer(
        msa_dir=msa_dir_train,
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes_tree_estimation=num_processes_tree_estimation,
        num_processes_counting=num_processes_counting,
        num_processes_optimization=num_processes_optimization_single_site,
        sites_subset_dir=contacting_sites_dir,
    )["learned_rate_matrix_path"]
    cherry_contact = read_rate_matrix(cherry_contact_path).to_numpy()
    del cherry_contact  # unused
    mutation_rate = compute_mutation_rate(read_rate_matrix(cherry_contact_path))
    print(
        f"***** cherry_contact_path = {cherry_contact_path} "
        f"({num_rate_categories} rate categories) with global mutation rate "
        f"{mutation_rate} *****"
    )

    cherry_contact_squared_path = os.path.join(
        chain_product_cached(
            rate_matrix_1_path=cherry_contact_path,
            rate_matrix_2_path=cherry_contact_path,
        )["output_rate_matrix_dir"],
        "result.txt",
    )

    # Now estimate and evaluate the coevolution model #
    cherry_2_path = coevolution_end_to_end_with_cherryml_optimizer(
        msa_dir=msa_dir_train,
        contact_map_dir=contact_map_dir_train,
        minimum_distance_for_nontrivial_contact=mdnc,
        coevolution_mask_path=get_aa_coevolution_mask_path(),
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes_tree_estimation=num_processes_tree_estimation,
        num_processes_counting=num_processes_counting,
        num_processes_optimization=num_processes_optimization_coevolution,
    )["learned_rate_matrix_path"]

    # Coevolution model without masking #
    cherry_2_no_mask_dir = coevolution_end_to_end_with_cherryml_optimizer(
        msa_dir=msa_dir_train,
        contact_map_dir=contact_map_dir_train,
        minimum_distance_for_nontrivial_contact=mdnc,
        coevolution_mask_path=None,
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes_tree_estimation=num_processes_tree_estimation,
        num_processes_counting=num_processes_counting,
        num_processes_optimization=num_processes_optimization_coevolution,
    )
    cherry_2_no_mask_path = cherry_2_no_mask_dir["learned_rate_matrix_path"]

    def get_runtime(lg_end_to_end_with_cherryml_optimizer_res: str) -> float:
        res = 0
        for lg_end_to_end_with_cherryml_optimizer_output_dir in [
            "count_matrices_dir_0",
            "jtt_ipw_dir_0",
            "rate_matrix_dir_0",
        ]:
            with open(
                os.path.join(
                    lg_end_to_end_with_cherryml_optimizer_res[
                        lg_end_to_end_with_cherryml_optimizer_output_dir
                    ],
                    "profiling.txt",
                ),
                "r",
            ) as profiling_file:
                profiling_file_contents = profiling_file.read()
                print(
                    f"{lg_end_to_end_with_cherryml_optimizer_output_dir} "  # noqa
                    f"profiling_file_contents = {profiling_file_contents}"  # noqa
                )
                res += float(profiling_file_contents.split()[2])
        return res

    runtime_coevolution = get_runtime(cherry_2_no_mask_dir)
    print(f"Runtime for coevolution: {runtime_coevolution}")

    contact_map_dir_test = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families_test,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes_tree_estimation,
    )["output_contact_map_dir"]
    contact_map_dir_test = create_maximal_matching_contact_map(
        i_contact_map_dir=contact_map_dir_test,
        families=families_test,
        minimum_distance_for_nontrivial_contact=mdnc,
        num_processes=num_processes_tree_estimation,
    )["o_contact_map_dir"]

    pair_site_rate_matrices = [
        (
            "Cherry contact squared",
            cherry_contact_squared_path,
        ),
        ("Cherry2", cherry_2_path),
        ("Cherry2; no mask", cherry_2_no_mask_path),
    ]

    for rate_matrix_2_name, rate_matrix_2_path in pair_site_rate_matrices:
        mutation_rate = compute_mutation_rate(
            read_rate_matrix(rate_matrix_2_path)
        )
        print(
            f"***** Evaluating: {rate_matrix_2_name} at {rate_matrix_2_path} ({num_rate_categories} rate categories) with global mutation rate {mutation_rate} *****"  # noqa
        )
        ll = evaluate_pair_site_model_on_held_out_msas(
            msa_dir=msa_dir_test,
            contact_map_dir=contact_map_dir_test,
            families=families_test,
            rate_matrix_1_path=cherry_path,
            rate_matrix_2_path=rate_matrix_2_path,
            num_processes=num_processes_tree_estimation,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=num_rate_categories,
            ),
        )
        print(f"ll for {rate_matrix_2_name} = {ll}")
        log_likelihoods.append((rate_matrix_2_name, ll))

    log_likelihoods = pd.DataFrame(log_likelihoods, columns=["model", "LL"])
    log_likelihoods.set_index(["model"], inplace=True)

    for baseline in [True, False]:
        plt.figure(figsize=(6.4, 4.8))
        xs = list(log_likelihoods.index)
        jtt_ll = log_likelihoods.loc["JTT", "LL"]
        if baseline:
            heights = log_likelihoods.LL - jtt_ll
        else:
            heights = -log_likelihoods.LL
        print(f"xs = {xs}")
        print(f"heights = {heights}")
        plt.bar(
            x=xs,
            height=heights,
        )
        ax = plt.gca()
        ax.yaxis.grid()
        plt.xticks(rotation=90)
        if baseline:
            if TITLES:
                plt.title(
                    "Results on Pfam 15K data\n(held-out log-Likelihood "
                    "improvement over JTT)"
                )
        else:
            if TITLES:
                plt.title(
                    "Results on Pfam 15K data\n(held-out negative log-Likelihood)"
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_image_dir,
                f"log_likelihoods_{num_rate_categories}_{baseline}",
            ),
            dpi=300,
        )
        plt.close()
    res = {
        "cherry_contact_squared_path": cherry_contact_squared_path,
        "cherry_2_no_mask_path": cherry_2_no_mask_path,
        "cherry_2_path": cherry_2_path,
    }
    return res


# Fig. 2a, 2b
def fig_pair_site_quantization_error(
    Q_2_name: str,
    num_rate_categories: int = 1,
    num_sequences: int = 1024,
    num_families_train: int = 15051,
    num_processes_tree_estimation: int = 32,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 8,
    angstrom_cutoff: float = 8.0,
    minimum_distance_for_nontrivial_contact: int = 7,
    random_seed_simulation: int = 0,
):
    output_image_dir = (
        f"images/fig_pair_site_quantization_error__Q_2_name__{Q_2_name}"
    )
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    qs = [
        (0.03, 1.1, 64),
    ]
    q_errors = [(np.sqrt(q[1]) - 1) * 100 for q in qs]
    q_points = [2 * q[2] + 1 for q in qs]
    yss_relative_errors = []
    Qs = []
    for (
        i,
        (
            quantization_grid_center,
            quantization_grid_step,
            quantization_grid_num_steps,
        ),
    ) in enumerate(qs):
        msg = f"***** grid = {(quantization_grid_center, quantization_grid_step, quantization_grid_num_steps)} *****"  # noqa
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR
        )
        families_train = families_all[:num_families_train]

        mdnc = minimum_distance_for_nontrivial_contact

        def get_nondiag_mask_matrix(Q):
            assert (Q != 0).all().all()
            assert Q.abs().min().min() > 1e-16
            Q = Q.to_numpy()
            res = np.ones(shape=Q.shape, dtype=int)
            for i in range(res.shape[0]):
                res[i, i] = 0
            return res

        def get_cotransitions_mask_matrix(Q):
            res = np.zeros(shape=Q.shape, dtype=int)
            for i, ab in enumerate(Q.index):
                for j, cd in enumerate(Q.index):
                    if ab[0] != cd[0] and ab[1] != cd[1]:
                        res[i, j] = 1
            return res

        def get_single_transitions_mask_matrix(Q):
            nondiag_mask_matrix = get_nondiag_mask_matrix(Q)
            cotransitions_mask_matrix = get_cotransitions_mask_matrix(Q)
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    if cotransitions_mask_matrix[i, j] == 1:
                        nondiag_mask_matrix[i, j] = 0
            return nondiag_mask_matrix

        if Q_2_name == "masked":
            Q_2_path = learn_coevolution_model_on_pfam15k()["cherry_2_path"]
            pi_2_path = os.path.join(
                get_stationary_distribution(rate_matrix_path=Q_2_path)[
                    "output_probability_distribution_dir"
                ],
                "result.txt",
            )
            coevolution_mask_path = "data/mask_matrices/aa_coevolution_mask.txt"
            mask_matrix = read_mask_matrix(coevolution_mask_path).to_numpy()
        elif Q_2_name.startswith("unmasked"):
            Q_2_path = learn_coevolution_model_on_pfam15k()[
                "cherry_2_no_mask_path"
            ]
            pi_2_path = os.path.join(
                get_stationary_distribution(rate_matrix_path=Q_2_path)[
                    "output_probability_distribution_dir"
                ],
                "result.txt",
            )
            coevolution_mask_path = None
            if Q_2_name == "unmasked-all-transitions":
                mask_matrix = get_nondiag_mask_matrix(
                    read_rate_matrix(Q_2_path)
                )
                assert mask_matrix.sum().sum() == 400 * 399
            elif Q_2_name == "unmasked-co-transitions":
                mask_matrix = get_cotransitions_mask_matrix(
                    read_rate_matrix(Q_2_path)
                )
                assert mask_matrix.sum().sum() == 400 * 19 * 19
            elif Q_2_name == "unmasked-single-transitions":
                mask_matrix = get_single_transitions_mask_matrix(
                    read_rate_matrix(Q_2_path)
                )
                assert mask_matrix.sum().sum() == 400 * 19 * 2

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_coevolution(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
            minimum_distance_for_nontrivial_contact=mdnc,
            angstrom_cutoff=angstrom_cutoff,
            num_sequences=num_sequences,
            families=families_all,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes_tree_estimation,
            random_seed=random_seed_simulation,
            pi_2_path=pi_2_path,
            Q_2_path=Q_2_path,
            use_cpp_simulation_implementation=True,
        )

        res_dict = coevolution_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            minimum_distance_for_nontrivial_contact=mdnc,
            coevolution_mask_path=coevolution_mask_path,
            families=families_train,
            tree_estimator=partial(
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
        )

        learned_rate_matrix_path = os.path.join(
            res_dict["rate_matrix_dir_0"],
            "result.txt",
        )

        learned_rate_matrix_df = read_rate_matrix(learned_rate_matrix_path)

        learned_rate_matrix = learned_rate_matrix_df.to_numpy()
        Qs.append(learned_rate_matrix)

        Q_2_df = read_rate_matrix(Q_2_path)
        Q_2 = Q_2_df.to_numpy()

        yss_relative_errors.append(
            relative_errors(
                Q_2,
                learned_rate_matrix,
                mask_matrix,
            )
        )

        print(f"Q_2_df.loc['VI', 'IV'] = {Q_2_df.loc['VI', 'IV']}")
        print(
            "learned_rate_matrix_df.loc['VI', 'IV'] = "
            f"{learned_rate_matrix_df.loc['VI', 'IV']}"
        )

    for i in range(len(q_points)):
        for density_plot in [False, True]:
            plot_rate_matrix_predictions(
                Q_2, Qs[i], mask_matrix, density_plot=density_plot
            )
            if TITLES:
                plt.title(
                    "True vs predicted rate matrix entries\nmax quantization error "
                    "= %.1f%% (%i quantization points)"
                    % (q_errors[i], q_points[i])
                )
            plt.tight_layout()
            plt.savefig(
                f"{output_image_dir}/log_log_plot_{i}_density_{density_plot}",
                dpi=300,
            )
            plt.close()

    df = pd.DataFrame(
        {
            "Quantization points": sum(
                [
                    [q_points[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "Relative error": sum(yss_relative_errors, []),
        }
    )
    df["Log relative error"] = np.log(df["Relative error"])

    sns.violinplot(
        x="Quantization points",
        y="Log relative error",
        data=df,
        inner=None,
        grid=False,
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as quantization improves",
        xlabel="Quantization points",
        grid=False,
    )

    plt.savefig(
        f"{output_image_dir}/violin_plot",
        dpi=300,
    )
    plt.close()


# Fig. 2c, 2d
def fig_coevolution_vs_indep():
    output_image_dir = f"images/fig_coevolution_vs_indep"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from numpy import linalg as la
    from scipy import stats
    from scipy.linalg import expm

    def plotHeatmap(
        inputMat,
        xlabel,
        xticklabels,
        ylabel,
        yticklabels,
        title="",
        plot_file="",
        font_color="w",
        figsize=(8, 8),
        round=2,
        **kwargs,
    ):

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(inputMat, **kwargs)
        # show ticks and label them
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_xticklabels(xticklabels, fontsize=14)
        ax.set_yticklabels(yticklabels, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        # rotate tick labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        # add text annotation to each cell
        for i in range(len(xticklabels)):
            for j in range(len(yticklabels)):
                text = ax.text(
                    i,
                    j,
                    np.round(inputMat[j, i], round),
                    ha="center",
                    va="center",
                    color=font_color,
                )
                # text = ax.text(i, j, inputMat[j, i], ha="center", va="center", color="w")

        # title
        if TITLES:
            ax.set_title(title, fontsize=14)
        fig.tight_layout()
        if len(plot_file) > 0:
            plt.savefig(plot_file, dpi=300)
        plt.show()

    def heatmap(
        data,
        row_labels,
        col_labels,
        title="",
        ax=None,
        cbar_kw={},
        cbarlabel="",
        **kwargs,
    ):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        mask = np.triu(np.ones_like(data, dtype=np.bool), k=1)
        data = np.ma.array(data, mask=mask)
        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)  # vmin, vmax
        ax.set_title(title, fontsize=18)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=18)

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels, fontsize=14)
        ax.set_yticklabels(row_labels, fontsize=14)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # boxes = []
        for i in range(10):
            ax.add_patch(
                patches.Rectangle(
                    (i - 0.46, i - 0.46),
                    0.92,
                    0.92,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
        # Add the patch to the Axes

        return im, cbar

    def dict2mat(dict, alphabet):
        mat = np.zeros((20, 20))
        for iI in range(20):
            for iJ in range(20):
                mat[iI, iJ] = dict[alphabet[iI] + alphabet[iJ]]
        return mat

    def exp_m_t(t, rate_mat, alphabet):
        probs = {}
        exp_mat = expm(t * rate_mat.to_numpy())
        for first_aa in alphabet:
            entries = [
                list(rate_mat.columns).index(first_aa + second_aa)
                for second_aa in alphabet
            ]
            probs[first_aa] = exp_mat[entries, :][:, entries]
        return probs

    def transition_prob(rate_mat):
        new_mat = rate_mat.copy()
        for state in rate_mat.columns:
            new_mat.loc[state] = new_mat.loc[state] / (
                -new_mat.loc[state, state]
            )
        return new_mat

    def off_diagonal(rate_mat, alphabet):
        probs = {}
        new_mat = rate_mat.copy()
        for state in rate_mat.columns:
            new_mat.loc[state] = new_mat.loc[state] / (
                -new_mat.loc[state, state]
            )
        for first_aa in alphabet:
            entries = [
                list(rate_mat.columns).index(first_aa + second_aa)
                for second_aa in alphabet
            ]
            probs[first_aa] = new_mat.to_numpy()[entries, :][:, entries]
        return probs

    def sort_by_val(dictionary, reverse=True):
        return {
            k: v
            for k, v in sorted(
                dictionary.items(), key=lambda item: item[1], reverse=reverse
            )
        }

    def sort_by_val_p(dictionary, reverse=True):
        total = np.sum(list(dictionary.values()))
        return {
            k: v / total
            for k, v in sorted(
                dictionary.items(), key=lambda item: item[1], reverse=reverse
            )
        }

    def select_by_val(dict, parent, threshold):
        return [
            [np.round(dict[child], 3), parent, child]
            for child in dict.keys()
            if np.abs(dict[child]) >= threshold
        ]

    by_hydropathy = [
        "I",
        "V",
        "L",
        "F",
        "C",
        "M",
        "A",
        "G",
        "T",
        "S",
        "W",
        "Y",
        "P",
        "H",
        "N",
        "Q",
        "D",
        "E",
        "K",
        "R",
    ]
    by_polarity = [
        "I",
        "V",
        "L",
        "F",
        "M",
        "A",
        "W",
        "P",
        "C",
        "G",
        "T",
        "S",
        "Y",
        "N",
        "Q",
        "H",
        "D",
        "E",
        "K",
        "R",
    ]

    rate_matrices_dict = learn_coevolution_model_on_pfam15k()

    Q1_contact_x_Q1_contact__1_rates_path = rate_matrices_dict[
        "cherry_contact_squared_path"
    ]
    Q2_mask__1_rates_path = rate_matrices_dict["cherry_2_path"]
    nomaskQ_path = rate_matrices_dict["cherry_2_no_mask_path"]

    productQ = pd.read_csv(
        Q1_contact_x_Q1_contact__1_rates_path, sep="\t", keep_default_na=False
    )
    maskQ = pd.read_csv(Q2_mask__1_rates_path, sep="\t", keep_default_na=False)
    nomaskQ = pd.read_csv(nomaskQ_path, sep="\t", keep_default_na=False)

    productQ = productQ.rename(columns={"Unnamed: 0": "state"}).set_index(
        "state"
    )
    maskQ = maskQ.rename(columns={"Unnamed: 0": "state"}).set_index("state")
    nomaskQ = nomaskQ.rename(columns={"Unnamed: 0": "state"}).set_index("state")

    val, vec = la.eig(np.transpose(productQ.to_numpy()))
    stationary_product_dict = dict(
        zip(
            productQ.columns.to_numpy(),
            np.round(
                vec.real[:, np.argmax(val.real)]
                / np.sum(vec.real[:, np.argmax(val.real)]),
                4,
            ),
        )
    )

    val, vec = la.eig(np.transpose(nomaskQ.to_numpy()))
    stationary_nomask_dict = dict(
        zip(
            nomaskQ.columns.to_numpy(),
            np.round(
                vec[:, np.argmax(val)] / np.sum(vec[:, np.argmax(val)]), 4
            ),
        )
    )

    val, vec = la.eig(np.transpose(maskQ.to_numpy()))
    stationary_mask_dict = dict(
        zip(
            maskQ.columns.to_numpy(),
            np.round(
                vec[:, np.argmax(val)] / np.sum(vec[:, np.argmax(val)]), 4
            ),
        )
    )

    vmin_val = np.min(
        (
            np.min(100 * dict2mat(stationary_product_dict, by_hydropathy)),
            np.min(100 * dict2mat(stationary_nomask_dict, by_hydropathy)),
        )
    )
    vmax_val = np.max(
        (
            np.max(100 * dict2mat(stationary_product_dict, by_hydropathy)),
            np.max(100 * dict2mat(stationary_nomask_dict, by_hydropathy)),
        )
    )

    # plt.rcParams.update({'font.size': 11})
    plotHeatmap(
        100 * dict2mat(stationary_product_dict, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="2-site Stationary Distribution under the Prouduct Model",
        figsize=(10, 10),
        vmin=vmin_val,
        vmax=vmax_val,
        plot_file=os.path.join(output_image_dir, "stationary_product"),
    )
    plotHeatmap(
        100 * dict2mat(stationary_nomask_dict, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="2-site Stationary Distribution under the noMask Model",
        figsize=(10, 10),
        vmin=vmin_val,
        vmax=vmax_val,
        plot_file=os.path.join(output_image_dir, "stationary_nomask"),
    )
    plotHeatmap(
        dict2mat(stationary_nomask_dict, by_hydropathy)
        / dict2mat(stationary_product_dict, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="noMask model stationary distribution / product model stationary distribution ",
        figsize=(10, 10),
        vmin=-0.5,
        vmax=2.5,
        font_color="black",
        cmap=plt.get_cmap("bwr"),
        plot_file=os.path.join(output_image_dir, "stationary_ratio"),
    )

    # Diagonal entries
    diagonal_product = {}
    for state in productQ.columns:
        diagonal_product[state] = np.round(-productQ[state][state], 2)

    diagonal_nomask = {}
    for state in nomaskQ.columns:
        diagonal_nomask[state] = np.round(-nomaskQ[state][state], 2)

    diagonal_mask = {}
    for state in maskQ.columns:
        diagonal_mask[state] = np.round(-maskQ[state][state], 2)

    # plt.rcParams.update({'font.size': 11})
    plotHeatmap(
        dict2mat(diagonal_product, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="2-site mutation rates in the Product Model",
        figsize=(10, 10),
        vmin=0.27,
        vmax=2.56,
        plot_file=os.path.join(output_image_dir, "diagonal_product"),
    )
    plotHeatmap(
        dict2mat(diagonal_nomask, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="2-site mutation rates in the noMask Model",
        figsize=(10, 10),
        vmin=0.27,
        vmax=2.56,
        plot_file=os.path.join(output_image_dir, "diagonal_nomask"),
    )
    plotHeatmap(
        dict2mat(diagonal_nomask, by_hydropathy)
        / dict2mat(diagonal_product, by_hydropathy),
        "Second amino acid",
        by_hydropathy,
        "First amino acid",
        by_hydropathy,
        title="noMask model diagonal / product model diagonal",
        figsize=(10, 10),
        vmin=0.23,
        vmax=1.77,
        font_color="black",
        cmap=plt.get_cmap("bwr"),
        plot_file=os.path.join(output_image_dir, "diagonal_ratio"),
    )


@caching.cached()
def get_site_rates_by_num_nontrivial_contacts(
    contact_map_dir: str,
    site_rates_dir: str,
    families: List[str],
    minimum_distance_for_nontrivial_contact: int,
):
    site_rates_by_num_nontrivial_contacts = defaultdict(list)
    for family in families:
        contact_map = read_contact_map(
            os.path.join(contact_map_dir, family + ".txt")
        )
        site_rates = read_site_rates(
            os.path.join(site_rates_dir, family + ".txt")
        )

        n = contact_map.shape[0]
        for i in range(n):
            num_nontrivial_contacts = 0
            for j in range(n):
                if (
                    abs(i - j) >= minimum_distance_for_nontrivial_contact
                    and contact_map[i, j] == 1
                ):
                    num_nontrivial_contacts += 1
            site_rates_by_num_nontrivial_contacts[
                num_nontrivial_contacts
            ].append(site_rates[i])
    return site_rates_by_num_nontrivial_contacts


# Fig. 2e
def fig_site_rates_vs_number_of_contacts(
    num_rate_categories: int = 20,
    num_sequences: int = 1024,
    num_families: int = 15051,
    num_processes: int = 32,
    angstrom_cutoff: float = 8.0,
    minimum_distance_for_nontrivial_contact: int = 7,
    fontsize: int = 14,
) -> Dict:
    """
    Returns a dictionary with the learned rate matrices.

    Test set can be used to compute held-out log-likelihoods.
    """
    output_image_dir = "images/fig_site_rates_vs_number_of_contacts"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    PFAM_15K_MSA_DIR = "input_data/a3m"
    PFAM_15K_PDB_DIR = "input_data/pdb"

    random_seed = 0

    families_all = get_families_pfam_15k(
        PFAM_15K_MSA_DIR,
    )
    np.random.seed(random_seed)
    np.random.shuffle(families_all)
    families = sorted(families_all[:num_families])

    # Subsample the MSAs
    msa_dir = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    tree_estimator = partial(
        fast_tree,
        num_rate_categories=num_rate_categories,
    )

    tree_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=get_lg_path(),
        num_processes=num_processes,
    )

    contact_map_dir = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes,
    )["output_contact_map_dir"]

    site_rates_by_num_nontrivial_contacts = get_site_rates_by_num_nontrivial_contacts(  # noqa
        contact_map_dir=contact_map_dir,
        site_rates_dir=tree_dirs["output_site_rates_dir"],
        families=families,
        minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,  # noqa
    )

    # num_contacts_list = list(range(1, 21, 1))
    # num_contacts_flattened = sum([len(site_rates_by_num_nontrivial_contacts[x]) * [x] for x in num_contacts_list], [])
    # site_rates_flattened = sum([site_rates_by_num_nontrivial_contacts[x] for x in num_contacts_list], [])

    # df = pd.DataFrame(
    #     {
    #         "Number of non-trivial contacts": num_contacts_flattened,
    #         "Site rate": site_rates_flattened,
    #     }
    # )

    # sns.violinplot(
    #     x="Number of non-trivial contacts",
    #     y="Site rate",
    #     data=df,
    #     inner=None,
    # )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xs = list(range(0, 19, 1))
    x_ticks = list(range(0, 19, 2))
    means = []
    medians = []
    upper_qt = []
    lower_qt = []
    number_of_sites = []
    for x in xs:
        means.append(np.mean(site_rates_by_num_nontrivial_contacts[x]))
        medians.append(np.median(site_rates_by_num_nontrivial_contacts[x]))
        upper_qt.append(
            np.quantile(site_rates_by_num_nontrivial_contacts[x], 0.75)
        )
        lower_qt.append(
            np.quantile(site_rates_by_num_nontrivial_contacts[x], 0.25)
        )
        number_of_sites.append(len(site_rates_by_num_nontrivial_contacts[x]))
    if TITLES:
        plt.title(
            "Site rate as a function of the number of non-trivial contacts"
        )
    plt.xlabel("Number of non-trivial contacts", fontsize=fontsize)
    plt.ylabel("Site rate", fontsize=fontsize)
    plt.xticks(x_ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.plot(xs, means, label="Mean site rate", color="r")
    # plt.plot(xs, medians, label="median site rate")
    ax.fill_between(
        xs,
        lower_qt,
        upper_qt,
        color="b",
        alpha=0.2,
        label="Interquartile range",
    )
    plt.grid()
    plt.legend(fontsize=fontsize)

    plt.savefig(
        f"{output_image_dir}/site_rate_vs_num_contacts",
        dpi=300,
    )
    plt.close()

    print(f"number_of_sites = {number_of_sites}")

    if TITLES:
        plt.title("Number of sites with a given number of non-trivial contacts")
    # plt.xlabel("number of non-trivial contacts")
    # plt.ylabel("number of sites")
    # plt.plot(xs, number_of_sites)
    # plt.savefig(
    #     f"{output_image_dir}/num_contacts_distribution",
    #     dpi=300,
    # )
    # plt.close()


# Comment in paragraph.
def fig_MSA_VI_cotransition(
    num_families_train: int = 10,
    aa_1: str = "E",
    aa_2: str = "K",
    families: List[str] = ["4kv7_1_A"],
    num_sequences: int = 1024,
):
    output_image_dir = "images/fig_pfam15k"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    PFAM_15K_MSA_DIR = "input_data/a3m"

    num_processes = 32
    train_test_split_seed = 0

    families_all = get_families_pfam_15k(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
    )
    np.random.seed(train_test_split_seed)
    np.random.shuffle(families_all)

    families_train = sorted(families_all[:num_families_train])

    # Subsample the MSAs
    msa_dir_train = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families_train,
        num_processes=num_processes,
    )["output_msa_dir"]

    for family in families_train:
        if families is not None and family not in families:
            continue
        msa_path = os.path.join(msa_dir_train, family + ".txt")
        print(f"MSA path = {msa_path}")
        msa = read_msa(msa_path)
        msa_list = list(msa.items())
        num_seqs = len(msa_list)
        seq_len = len(msa_list[0][1])
        position_aas = [set() for _ in range(seq_len)]
        for i in range(seq_len):
            for n in range(num_seqs):
                position_aas[i].add(msa_list[n][1][i])
        cols_with_IV = []
        for i in range(seq_len):
            if aa_1 in position_aas[i] and aa_2 in position_aas[i]:
                cols_with_IV.append(i)

        for i in cols_with_IV:
            for j in cols_with_IV:
                if i >= j:
                    continue
                IVs = []
                VIs = []
                IIs = []
                VVs = []
                for n in range(num_seqs):
                    if msa_list[n][1][i] == aa_1 and msa_list[n][1][j] == aa_2:
                        IVs.append(n)
                    elif (
                        msa_list[n][1][i] == aa_2 and msa_list[n][1][j] == aa_1
                    ):
                        VIs.append(n)
                    elif (
                        msa_list[n][1][i] == aa_1 and msa_list[n][1][j] == aa_1
                    ):
                        IIs.append(n)
                    elif (
                        msa_list[n][1][i] == aa_2 and msa_list[n][1][j] == aa_2
                    ):
                        VVs.append(n)
                tot_IV_VI_II_VV = len(IVs) + len(VIs) + len(IIs) + len(VVs)
                if tot_IV_VI_II_VV >= num_seqs / 8:
                    pct_IV = len(IVs) / tot_IV_VI_II_VV
                    pct_VI = len(VIs) / tot_IV_VI_II_VV
                    pct_II = len(IIs) / tot_IV_VI_II_VV
                    pct_VV = len(VVs) / tot_IV_VI_II_VV
                    if pct_IV > 0.2 and pct_VI > 0.2:
                        print(
                            f"sites ({i}, {j}): ({aa_1}{aa_2}, {aa_2}{aa_1}, "
                            f"{aa_1}{aa_1}, {aa_2}{aa_2}) = "
                            "(%.3f, %.3f, %.3f, %.3f)"
                            % (
                                pct_IV,
                                pct_VI,
                                pct_II,
                                pct_VV,
                            ),
                            f"over {tot_IV_VI_II_VV} pairs",
                        )
