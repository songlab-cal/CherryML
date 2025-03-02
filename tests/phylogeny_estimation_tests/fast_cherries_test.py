import tempfile
import unittest
from cherryml.phylogeny_estimation._fast_cherries import fast_cherries
from cherryml.markov_chain import get_equ_path, get_lg_path, normalized
from cherryml.benchmarking.lg_paper import run_rate_estimator
from cherryml.io import read_rate_matrix, read_msa, read_tree, read_site_rates
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from cherryml.utils import get_families
from cherryml.config import create_config_from_dict
from cherryml import caching
import pandas as pd
from cherryml import lg_end_to_end_with_cherryml_optimizer
from cherryml.phylogeny_estimation.phylogeny_estimator import get_phylogeny_estimator_from_config
from cherryml.estimation_end_to_end import CHERRYML_TYPE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytest


class TestFastCherries(unittest.TestCase):
    def test_run_fast_cherries(self):
        """
        just runs the solver to see it runs
        """  
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_trees")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates")

        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_trees", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred whilegi removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        # WAG
        fast_cherries(
            msa_dir = "tests/data",
            families = [f[:-4] for f in listdir("tests/data")[:400] if isfile(join("tests/data", f)) and f[-4:] == ".txt"],
            rate_matrix_path = get_equ_path(),
            num_rate_categories= 1,
            max_iters = 50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )
        # WAG
        fast_cherries(
            msa_dir = "tests/data",
            families = [f[:-4] for f in listdir("tests/data")[:400] if isfile(join("tests/data", f)) and f[-4:] == ".txt"],
            rate_matrix_path = get_equ_path(),
            num_rate_categories= 1,
            max_iters = 50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )
        # LG
        fast_cherries(
            msa_dir = "tests/data",
            families = [f[:-4] for f in listdir("tests/data")[:400] if isfile(join("tests/data", f)) and f[-4:] == ".txt"],
            rate_matrix_path = get_equ_path(),
            num_rate_categories= 20,
            max_iters = 50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )
    def test_run_different_alphabet(self):
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_trees")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates")

        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_trees", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred whilegi removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        # LG
        fast_cherries(
            msa_dir = "tests/phylogeny_estimation_tests/different_alphabet",
            families = ["msa"],
            rate_matrix_path = "tests/phylogeny_estimation_tests/weird_rate_matrix.txt",
            num_rate_categories= 4,
            max_iters = 50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )

        fast_cherries_output_path = "tests/phylogeny_estimation_tests/pairing_ble_test_trees/msa.txt"
        tree = read_tree(fast_cherries_output_path)
        cherries = tree.children(tree.root()) 
        assert(len(cherries) == 2)
        expected = [("s1", "s3"),("s3", "s1"),("s4", "s2"),("s2", "s4")] 
        
        cherry_with_lengths = tree.children(cherries[0][0])
        assert((cherry_with_lengths[0][0],cherry_with_lengths[1][0]) in expected)

        cherry_with_lengths = tree.children(cherries[1][0])
        assert((cherry_with_lengths[0][0],cherry_with_lengths[1][0]) in expected)
        

    @pytest.mark.slow
    def test_correct_runtime(self):
        """
        ensures that the runtimes are being recorded correctly
        """  
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_trees")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates")

        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_trees", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred whilegi removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        
        # LG
        fast_cherries(
            msa_dir = "tests/phylogeny_estimation_tests/timing_data",
            families = ["large_msa", "smol_msa"],
            rate_matrix_path = get_equ_path(),
            num_rate_categories= 20,
            max_iters=50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )
        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/large_msa.profiling", "r") as f:
            large_total = float(f.readlines()[3].split()[-1])
        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/smol_msa.profiling", "r") as f:
            small_total = float(f.readlines()[3].split()[-1])
        assert large_total > small_total


        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/large_msa.profiling", "r") as f:
            large_cpp = float(f.readlines()[2].split()[-1])
        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/smol_msa.profiling", "r") as f:
            small_cpp = float(f.readlines()[2].split()[-1])
        assert large_cpp > small_cpp

        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/large_msa.profiling", "r") as f:
            large_ble = float(f.readlines()[1].split()[-1])
        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/smol_msa.profiling", "r") as f:
            small_ble = float(f.readlines()[1].split()[-1])
        assert large_ble > small_ble

        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/large_msa.profiling", "r") as f:
            large_pairing = float(f.readlines()[0].split()[-1])
        with open("tests/phylogeny_estimation_tests/pairing_ble_test_trees/smol_msa.profiling", "r") as f:
            small_pairing = float(f.readlines()[0].split()[-1])
        assert large_pairing > small_pairing

        assert large_pairing + large_ble < large_cpp
        assert small_pairing + small_ble < small_cpp
    
    def test_correct_wrapper(self):
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_trees")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods")
        if not os.path.exists("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            os.makedirs("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates")

        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_trees"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_trees", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        for filename in os.listdir("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates"):
            file_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_site_rates", filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error occurred while removing {file_path}: {e}")
        # LG
        fast_cherries(
            msa_dir = "tests/data",
            families = get_families("tests/data"),
            rate_matrix_path = get_equ_path(),
            num_rate_categories= 1,
            max_iters = 50,
            num_processes=1,
            remake=False,
            output_tree_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_trees",
            output_site_rates_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_site_rates",
            output_likelihood_dir = "tests/phylogeny_estimation_tests/pairing_ble_test_likelihoods",
        )

        for family in get_families("tests/data"):
            msa_path = os.path.join("tests/data", family + ".txt")
            msa = read_msa(msa_path)

            fast_cherries_output_path = os.path.join("tests/phylogeny_estimation_tests/pairing_ble_test_trees", family + ".txt")
            tree = read_tree(fast_cherries_output_path)
            assert len(tree.children(tree.root())) == len(msa)//2 + len(msa)%2

    def test_multiprocessing(self):
        """
        Makes sure that using multiple processes gives the same result as using one process.
        """
        families = [f[:-4] for f in listdir("tests/data")[:400] if isfile(join("tests/data", f)) and f[-4:] == ".txt"]
        with tempfile.TemporaryDirectory() as cache_one_process:
            with tempfile.TemporaryDirectory() as cache_multiple_processes:
                caching.set_cache_dir(cache_one_process)
                output_dirs_one_process = fast_cherries(
                    msa_dir = "tests/data",
                    families = families,
                    rate_matrix_path = get_equ_path(),
                    num_rate_categories= 20,
                    max_iters = 50,
                    num_processes=1,
                    remake=False,
                )

                caching.set_cache_dir(cache_multiple_processes)
                output_dirs_multiple_processes = fast_cherries(
                    msa_dir = "tests/data",
                    families = families,
                    rate_matrix_path = get_equ_path(),
                    num_rate_categories= 20,
                    max_iters = 50,
                    num_processes=2,
                    remake=False,
                )
                failed_families = []
                ok_families = []
                for i, family in enumerate(families):
                    # tree_one_process = read_tree(os.path.join(output_dirs_one_process["output_tree_dir"], family + ".txt"))
                    # tree_multiple_processes = read_tree(os.path.join(output_dirs_multiple_processes["output_tree_dir"], family + ".txt"))
                    # if tree_one_process.to_newick(format=2) != tree_multiple_processes.to_newick(format=2):
                    #     failed_families.append(i)
                    site_rates_one_process = read_site_rates(os.path.join(output_dirs_one_process["output_site_rates_dir"], family + ".txt"))
                    site_rates_multiple_processes = read_site_rates(os.path.join(output_dirs_multiple_processes["output_site_rates_dir"], family + ".txt"))
                    if site_rates_one_process != site_rates_multiple_processes:
                        failed_families.append(i)
                    else:
                        ok_families.append(i)
                if len(failed_families) > 0:
                    raise Exception(f"The outputs for the following families differ: {failed_families}. These are OK: {ok_families}")
