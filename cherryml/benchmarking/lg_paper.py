"""
Reproduce LG paper results.

This module should be run from the repo root directory.
"""
import logging
import os
import sys
import tempfile
import time
from functools import partial
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wget

from cherryml.config import Config, create_config_from_dict
from cherryml import (
    PhylogenyEstimatorType,
    lg_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_em_optimizer,
)
from cherryml.estimation_end_to_end import CHERRYML_TYPE
from cherryml.io import read_log_likelihood
from cherryml.markov_chain import (
    get_equ_path,
    get_jtt_path,
    get_lg_path,
    get_wag_path,
)
from cherryml.phylogeny_estimation.phylogeny_estimator import get_phylogeny_estimator_from_config
from cherryml.utils import pushd

from .globals import IMG_EXTENSIONS


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


init_logger()
logger = logging.getLogger(__name__)


def verify_integrity(filepath: str, mode: str = "555"):
    if not os.path.exists(filepath):
        logger.error(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
    mask = oct(os.stat(filepath).st_mode)[-3:]
    if mask != mode:
        logger.error(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )
        raise Exception(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )


def verify_integrity_of_directory(
    dirpath: str, expected_number_of_files: int, mode: str = "555"
):
    """
    Makes sure that the directory has the expected number of files and that
    they are all write protected (or another specified mode).
    """
    dirpath = os.path.abspath(dirpath)
    if not os.path.exists(dirpath):
        logger.error(
            f"Trying to verify the integrity of an inexistent "
            f"directory: {dirpath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent "
            f"diretory: {dirpath}"
        )
    filenames = sorted(list(os.listdir(dirpath)))
    if len(filenames) != expected_number_of_files:
        raise Exception(
            f"{dirpath} already exists but does not contain the "
            "expected_number_of_files."
            f"\nExpected: {expected_number_of_files}\nFound: {len(filenames)}"
        )
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        verify_integrity(filepath=filepath, mode=mode)


def wget_tarred_data_and_chmod(
    url: str,
    destination_directory: str,
    expected_number_of_files: int,
    mode: str = "555",
) -> None:
    """
    Download tar data from a url if not already present.

    Gets tarred data from `url` into `destination_directory` and chmods the
    data to 555 (or the `mode` specified) so that it is write protected.
    `expected_number_of_files` is the expected number of files after untarring.
    If the data is already present (which is determined by seeing whether the
    expected_number_of_files match), then the data is not downloaded again.

    Args:
        url: The url of the tar data.
        destination_directory: Where to untar the data to.
        expected_number_of_files: The expected number of files after
            untarring.
        mode: What mode to change the files to.

    Raises:
        Exception if the expected_number_of_files are not found after untarring,
            or if the data fails to download, etc.
    """
    destination_directory = os.path.abspath(destination_directory)
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=expected_number_of_files,
            mode=mode,
        )
        logger.info(
            f"{url} has already been downloaded successfully to "
            f"{destination_directory}. Not downloading again."
        )
        return
    logger.info(f"Downloading {url} into {destination_directory}")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    logger.info(f"pushd into {destination_directory}")
    with pushd(destination_directory):
        wget.download(url)
        logger.info(f"wget {url} into {destination_directory}")
        os.system("tar -xvzf *.tar.gz >/dev/null")
        logger.info("Untarring file ...")
        os.system("rm *.tar.gz")
        logger.info("Removing tar file ...")
    os.system(f"chmod -R {mode} {destination_directory}")
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=expected_number_of_files,
        mode=mode,
    )
    logger.info("Success!")


def _convert_lg_data(
    lg_data_dir: str,
    destination_directory: str,
) -> None:
    """
    Convert the LG MSAs from the PHYLIP format to our format.

    Args:
        lg_training_data_dir: Where the MSAs in PHYLIP format are.
        destination_directory: Where to write the converted MSAs to.
    """
    logger.info("Converting LG Training data to our MSA training format ...")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    protein_family_names = sorted(list(os.listdir(lg_data_dir)))
    for protein_family_name in protein_family_names:
        with open(os.path.join(lg_data_dir, protein_family_name), "r") as file:
            res = ""
            lines = file.read().split("\n")
            n_seqs, n_sites = map(int, lines[0].split(" "))
            for i in range(n_seqs):
                line = lines[2 + i]
                try:
                    protein_name, protein_sequence = line.split()
                except Exception:
                    raise ValueError(
                        f"For protein family {protein_family_name} , could "
                        f"not split line: {line}"
                    )
                assert len(protein_sequence) == n_sites
                res += f">{protein_name}\n"
                res += f"{protein_sequence}\n"
            output_filename = os.path.join(
                destination_directory,
                protein_family_name.replace(".", "_") + ".txt",
            )
            with open(output_filename, "w") as outfile:
                outfile.write(res)
                outfile.flush()
            os.system(f"chmod 555 {output_filename}")


def get_lg_PfamTestingAlignments_data(
    destination_directory: str,
) -> None:
    """
    Download the lg_PfamTestingAlignments data

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to download the data to.
    """
    url = (
        "http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_PfamTestingAlignments.tar.gz"
    )
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=500,
            mode="555",
        )
        logger.info(
            f"{url} has already been downloaded successfully "
            f"to {destination_directory}. Not downloading again."
        )
        return
    with tempfile.TemporaryDirectory() as destination_directory_unprocessed:
        wget_tarred_data_and_chmod(
            url=url,
            destination_directory=destination_directory_unprocessed,
            expected_number_of_files=500,
            mode="777",
        )
        _convert_lg_data(
            lg_data_dir=destination_directory_unprocessed,
            destination_directory=destination_directory,
        )
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=500,
        mode="555",
    )


def get_lg_PfamTrainingAlignments_data(
    destination_directory: str,
) -> None:
    """
    Get the lg_PfamTrainingAlignments.

    Downloads the lg_PfamTrainingAlignments data to the specified
    `destination_directory`, *converting it to our training MSA format in the
    process*.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to store the (converted) MSAs.
    """
    url = (
        "http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_PfamTrainingAlignments.tar.gz"
    )
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=3912,
            mode="555",
        )
        logger.info(
            f"{url} has already been downloaded successfully "
            f"to {destination_directory}. Not downloading again."
        )
        return
    with tempfile.TemporaryDirectory() as destination_directory_unprocessed:
        wget_tarred_data_and_chmod(
            url=url,
            destination_directory=destination_directory_unprocessed,
            expected_number_of_files=1,
            mode="777",
        )
        _convert_lg_data(
            lg_data_dir=os.path.join(
                destination_directory_unprocessed, "AllData"
            ),
            destination_directory=destination_directory,
        )
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=3912,
        mode="555",
    )


# @caching.cached()
def run_rate_estimator(
    rate_estimator_name: str,
    phylogeny_estimator_configs: Config,
    msa_train_dir: str,
    families_train: List[str],
    num_processes: int,
) -> str:
    """
    Given a rate estimator name, returns the path to the rate matrix
    """
    if rate_estimator_name == "EQU":
        res = get_equ_path()
    elif rate_estimator_name == "reproduced JTT":
        res = get_jtt_path()
    elif rate_estimator_name == "reproduced WAG":
        res = get_wag_path()
    elif rate_estimator_name == "reproduced LG":
        res = get_lg_path()
    elif rate_estimator_name.startswith("Cherry__"):
        tokens = rate_estimator_name.split("__")
        assert len(tokens) == 2
        res_dict = lg_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_train_dir,
            families=families_train,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=4,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_iterations=int(tokens[1]),
            num_processes_tree_estimation=num_processes,
            num_processes_counting=1,
            num_processes_optimization=1,
            edge_or_cherry="cherry",
        )
        with open(
            "lg_paper_fig__" + rate_estimator_name + "__profiling_str.txt", "w"
        ) as profiling_file:
            profiling_file.write(f"{res_dict['profiling_str']}")
        res = res_dict["learned_rate_matrix_path"]
        return res
    elif rate_estimator_name.startswith("Cherry++__"):
        tokens = rate_estimator_name.split("__")
        res_dict = lg_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_train_dir,
            families=families_train,
            tree_estimator=get_phylogeny_estimator_from_config(phylogeny_estimator_configs),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_iterations=int(tokens[1]),
            num_processes_tree_estimation=num_processes,
            num_processes_counting=1,
            num_processes_optimization=1,
            edge_or_cherry=CHERRYML_TYPE,
        )
        with open(
            "lg_paper_fig__" + rate_estimator_name + "__profiling_str.txt", "w"
        ) as profiling_file:
            profiling_file.write(f"{res_dict['profiling_str']}")
        res = res_dict["learned_rate_matrix_path"]
        return res
    elif rate_estimator_name.startswith("EM_FT__"):
        tokens = rate_estimator_name.split("__")
        assert len(tokens) == 3
        res_dict = lg_end_to_end_with_em_optimizer(
            msa_dir=msa_train_dir,
            families=families_train,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=4,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_iterations=int(tokens[1]),
            num_processes_tree_estimation=num_processes,
            num_processes_counting=1,
            num_processes_optimization=1,
            em_backend="xrate",
            extra_em_command_line_args=f"-log 6 -f 3 -mi {tokens[2]}",
        )
        res = res_dict["learned_rate_matrix_path"]
        with open(
            "lg_paper_fig__" + rate_estimator_name + "__profiling_str.txt", "w"
        ) as profiling_file:
            profiling_file.write(f"{res_dict['profiling_str']}")
        return res
    else:
        raise ValueError(f"Unknown rate estimator name: {rate_estimator_name}")
    return res


def get_reported_results_df(pfam_or_treebase: str) -> pd.DataFrame:
    """
    Gets the results table of the LG paper.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        pfam_or_treebase: 'pfam' or 'treebase'.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pfam_or_treebase == "treebase":
        df = pd.read_csv(
            os.path.join(dir_path, "data/lg_paper/Treebase.txt"), sep="\t"
        )
    elif pfam_or_treebase == "pfam":
        df = pd.read_csv(
            os.path.join(dir_path, "data/lg_paper/Pfam.txt"), sep="\t"
        )
    else:
        raise ValueError(
            f"pfam_or_treebase should be either 'pfam' or "
            f"'treebase'. You provided: {pfam_or_treebase}"
        )
    df = df.drop(0)
    df.set_index(["Name"], inplace=True)
    return df


def reproduce_lg_paper_fig_4(
    msa_train_dir: str,
    families_train: List[str],
    msa_test_dir: str,
    families_test: List[str],
    rate_estimator_names: List[Tuple[str]],
    phylogeny_estimator_configs:Config,
    baseline_rate_estimator_name: Optional[Tuple[str, str]],
    evaluation_phylogeny_estimator: PhylogenyEstimatorType,
    num_processes: int,
    pfam_or_treebase: str,
    family_name_len: int,
    figsize: Tuple[float, float] = (6.4, 4.8),
    num_bootstraps: int = 0,
    use_colors: bool = True,
    output_image_dir: str = "./",
    fontsize: int = 13,
):
    """
    Reproduce Fig. 4 of the LG paper, extending it with the desired models.
    """
    Qs = {}
    assert pfam_or_treebase == "pfam"
    assert family_name_len == 7
    df = pd.DataFrame(
        np.zeros(
            shape=(
                len(families_test),
                len(rate_estimator_names),
            )
        ),
        index=families_test,
        columns=rate_estimator_names,
    )

    reported_results_df = get_reported_results_df(
        pfam_or_treebase=pfam_or_treebase
    )

    df["num_sites"] = 0
    for family in families_test:
        df.loc[family, "num_sites"] = reported_results_df.loc[
            family[:family_name_len], "Sites"
        ]

    if baseline_rate_estimator_name is not None:
        rate_estimator_names_w_baseline = [
            baseline_rate_estimator_name
        ] + rate_estimator_names
        phylogeny_estimator_configs = [
            create_config_from_dict({"identifier":{}, "args":{}})
        ] + phylogeny_estimator_configs
    else:
        rate_estimator_names_w_baseline = list(set(rate_estimator_names))
    assert len(rate_estimator_names_w_baseline) == len(phylogeny_estimator_configs), f"there must be the same number of rate estimator configs as there are names! {len(rate_estimator_names_w_baseline)} {len(phylogeny_estimator_configs)}"
    using_lg = False
    for rate_estimator_name, config in zip(rate_estimator_names_w_baseline, phylogeny_estimator_configs):
        rate_estimator_name = rate_estimator_name[0]
        print(f"Evaluating rate_estimator_name: {rate_estimator_name}")
        st = time.time()
        if rate_estimator_name.startswith("reported"):
            _, rate_matrix_name = rate_estimator_name.split(" ")
            for family in families_test:
                df.loc[family, rate_estimator_name] = reported_results_df.loc[
                    family[:family_name_len], rate_matrix_name
                ]
        elif rate_estimator_name.startswith("path__"):
            rate_matrix_path = rate_estimator_name[6:]
        else:
            rate_matrix_path = run_rate_estimator(
                rate_estimator_name=rate_estimator_name,
                phylogeny_estimator_configs=config,
                msa_train_dir=msa_train_dir,
                families_train=families_train,
                num_processes=num_processes,
            )
            Qs[rate_estimator_name] = rate_matrix_path
            output_likelihood_dir = evaluation_phylogeny_estimator(
                msa_dir=msa_test_dir,
                families=families_test,
                rate_matrix_path=rate_matrix_path,
            )["output_likelihood_dir"]
            for family in families_test:
                metric = read_log_likelihood(
                    os.path.join(output_likelihood_dir, family + ".txt")
                )[0]
                df.loc[family, rate_estimator_name] = metric
        print(
            f"Total time to evaluate {rate_estimator_name} = {time.time() - st}"
        )

    def get_log_likelihoods(df: pd.DataFrame, model_names: List[str]):
        """
        Given a DataFrame like the LG results table, with Name as the index,
        returns the sum of log likelihoods for each model.
        """
        num_sites = df["num_sites"].sum()
        log_likelihoods = 2.0 * df[model_names].sum(axis=0) / num_sites
        if baseline_rate_estimator_name is not None:
            log_likelihoods -= (
                2.0 * df[baseline_rate_estimator_name[0]].sum() / num_sites
            )
        return log_likelihoods

    model_names = [x[0] for x in rate_estimator_names]
    y = get_log_likelihoods(df, model_names)
    if num_bootstraps > 0:
        np.random.seed(0)
        y_bootstraps = []
        for _ in range(num_bootstraps):
            chosen_rows = np.random.choice(
                df.index,
                size=len(df.index),
                replace=True,
            )
            df_bootstrap = df.loc[chosen_rows]
            assert df_bootstrap.shape == df.shape
            y_bootstrap = get_log_likelihoods(
                df_bootstrap, [x[0] for x in rate_estimator_names]
            )
            y_bootstraps.append(y_bootstrap)
        y_bootstraps = np.array(y_bootstraps)
        assert y_bootstraps.shape == (num_bootstraps, len(rate_estimator_names))

    colors = []
    for model_name in model_names:
        if not use_colors:
            colors.append("black")
        elif "reproduced" in model_name:
            colors.append("black")
        elif "FastTree" in model_name:
            colors.append("red")
        elif "Cherry" in model_name:
            colors.append("blue")
        elif "EM" in model_name:
            colors.append("yellow")
        else:
            colors.append("brown")
    plt.figure(figsize=figsize)
    pairing_times = []
    ble_times = []
    estimation_times = []
    total_times = []
    for x in rate_estimator_names:
        profiling_str = f"lg_paper_fig__{x[0]}__profiling_str.txt"
        found_pairing = False
        found_ble = False
        found_estimation = False
        found_total = False
        if os.path.isfile(profiling_str):
            with open(profiling_str, "r") as file:
                for line in file:
                    tokens = line.split(" ")
                    if tokens[0] == "time_tree_estimation":
                        t = float(tokens[-1])
                        estimation_times.append(t)
                        found_estimation = True
                    elif tokens[0] == "total_cpu_time:":
                        t = float(tokens[-1])
                        total_times.append(t)
                        found_total = True
                    
        if not found_estimation:
            estimation_times.append(0)
        if not found_pairing:
            pairing_times.append(0)
        if not found_ble:
            ble_times.append(0)
        if not found_total:
            total_times.append(0)

    plt.bar(
        x=[f"{x[1]}" for x in rate_estimator_names],
        height=y,
        color=colors,
    )
    plt.xticks(rotation=0, fontsize=fontsize)
    ax = plt.gca()
    ax.yaxis.grid()
    handles = [
        mpatches.Patch(color="black", label="Reproduced"),
        mpatches.Patch(color="red", label="FastTree"),
        mpatches.Patch(color="blue", label="FastCherries"),
    ]
    if any(["EM" in model_name for model_name in model_names]):
        handles.append(mpatches.Patch(color="yellow", label="EM"))
    if use_colors:
        plt.legend(
            handles=handles,
            fontsize=fontsize,
        )
    plt.tight_layout()
    if baseline_rate_estimator_name is not None:
        plt.ylabel(
            "Average per-site AIC\nimprovement over "
            f"{baseline_rate_estimator_name[1]}, in nats",
            fontsize=fontsize,
        )
    else:
        plt.ylabel("Average per-site AIC, in nats", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for IMG_EXTENSION in IMG_EXTENSIONS:
        plt.savefig(
            f"{output_image_dir}/lg_paper_figure{IMG_EXTENSION}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()

    plt.bar(x=[f"{x[1]}" for x in rate_estimator_names[2:]], 
            height=np.array(estimation_times[2:]) - np.array(ble_times[2:]) - np.array(pairing_times[2:]), 
            bottom=np.array(ble_times[2:]) + np.array(pairing_times[2:]), 
            label='Tree Estimation'
    )
    plt.bar(x=[f"{x[1]}" for x in rate_estimator_names[2:]], 
            height=np.array(total_times[2:]) - np.array(estimation_times[2:]), 
            bottom=np.array(estimation_times[2:]), 
            label='Rate Matrix Estimation'
    )
    plt.yticks(fontsize=fontsize)

    plt.ylabel('Runtime (s)', fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.xticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    img_path = f"runtime_comparison"  # noqa
    for IMG_EXTENSION in IMG_EXTENSIONS:
        plt.savefig(
            os.path.join(
                output_image_dir,
                img_path + f"{IMG_EXTENSION}",
            ),
            dpi=300,
        )
    plt.close()

    if num_bootstraps:
        return (
            y,
            df,
            pd.DataFrame(
                y_bootstraps, columns=[x[1] for x in rate_estimator_names]
            ),
            Qs,
        )
    else:
        return y, df, None, Qs
