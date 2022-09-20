import os
from typing import List, Optional
from cherryml.caching import cached_parallel_computation, cached_computation, set_use_hash, set_cache_dir, set_log_level


def test_cached_parallel_computation():
    @cached_parallel_computation(
        parallel_arg="families",
        exclude_args=["num_processes"],
        output_dirs=["output_msa_dir", "output_msa_dir2"],
    )
    def simulate_msas(
        tree_dir: str,
        site_rates_dir: str,
        contact_map_dir: str,
        families: List[str],
        amino_acids: List[str],
        pi_1_path: str,
        Q_1_path: str,
        pi_2_path: str,
        Q_2_path: str,
        strategy: str,
        num_processes: int,
        output_msa_dir: Optional[str] = None,
        output_msa_dir2: Optional[str] = None,  # Can also include a log directory? Or just write '.log' files in the only output directories that matter! (I think the latter is better)
    ):
        for family in families:
            output_filepath = os.path.join(output_msa_dir, family + ".txt")
            with open(output_filepath, "w") as f:
                f.write("Simulated MSA for family: " + family + "\n")
        for family in families:
            output_filepath = os.path.join(output_msa_dir2, family + ".txt")
            with open(output_filepath, "w") as f:
                f.write("Simulated MSA for family: " + family + "\n")

    set_use_hash(True)
    set_cache_dir("cache")
    set_log_level(9)

    # This fails because the output dirs - if passed - must be passed in as keywords.
    res = simulate_msas(
        tree_dir="path/to/tree_dir",
        site_rates_dir="path/to/site_rates_dir",
        contact_map_dir="path/to/contact_map_dir",
        families=["globin", "GPCR"],
        amino_acids=["T", "P"],
        pi_1_path="path/to/pi_1.txt",
        Q_1_path="path/to/Q_1.txt",
        pi_2_path="path/to/pi_2.txt",
        Q_2_path="path/to/Q_2.txt",
        strategy="jump_chain",
        # output_msa_dir="output_msa_dir",  # When commented, the cached directory is used. When provided, this is used
        # output_msa_dir2="output_msa_dir2",  # When commented, the cached directory is used. When provided, this is used
        num_processes=32,
    )
    print(res)


def test_cached_computation():
    @cached_computation(
        exclude_args=["num_processes"],
        output_dirs=["output_count_dir", "output_count_dir2"],
    )
    def count_data(
        tree_dir: str,
        site_rates_dir: str,
        contact_map_dir: str,
        families: List[str],
        amino_acids: List[str],
        pi_1_path: str,
        Q_1_path: str,
        pi_2_path: str,
        Q_2_path: str,
        strategy: str,
        num_processes: int,
        output_count_dir: Optional[str] = None,
        output_count_dir2: Optional[str] = None,  # Can also include a log directory? Or just write '.log' files in the only output directories that matter! (I think the latter is better)
    ):
        output_filepath = os.path.join(output_count_dir, "result.txt")
        with open(output_filepath, "w") as f:
            f.write("Counted data from families: " + str(families) + "\n")
        output_filepath = os.path.join(output_count_dir2, "result.txt")
        with open(output_filepath, "w") as f:
            f.write("Counted data from families: " + str(families) + "\n")

    set_use_hash(True)
    set_cache_dir("cache")
    set_log_level(9)

    # This fails because the output dirs - if passed - must be passed in as keywords.
    res = count_data(
        tree_dir="path/to/tree_dir",
        site_rates_dir="path/to/site_rates_dir",
        contact_map_dir="path/to/contact_map_dir",
        families=["globin", "GPCR"],
        amino_acids=["T", "P"],
        pi_1_path="path/to/pi_1.txt",
        Q_1_path="path/to/Q_1.txt",
        pi_2_path="path/to/pi_2.txt",
        Q_2_path="path/to/Q_2.txt",
        strategy="jump_chain",
        # output_count_dir="output_count_dir",  # When commented, the cached directory is used. When provided, this is used
        # output_count_dir2="output_count_dir2",  # When commented, the cached directory is used. When provided, this is used
        num_processes=32,
    )
    print(res)


if __name__ == "__main__":
    test_cached_parallel_computation()
    test_cached_computation()
