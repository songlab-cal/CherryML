import unittest

from cherryml.io import get_msa_num_sites


class Test_get_msa_num_sites(unittest.TestCase):
    def test_get_msa_num_sites(self):
        num_sites = get_msa_num_sites(
            "tests/test_input_data/a3m_small/1e7l_1_A.txt"
        )
        assert num_sites == 157

        num_sites = get_msa_num_sites(
            "tests/estimation_tests/test_input_data/msa_dir/fam1.txt"
        )
        assert num_sites == 4
