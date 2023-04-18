import unittest

from cherryml.io import (
    get_msa_num_residues,
    get_msa_num_sequences,
    get_msa_num_sites,
)


class Test_get_msa_num_sites(unittest.TestCase):
    def test(self):
        res = get_msa_num_sites("./tests/io_tests/1e7l_1_A.txt")
        self.assertEquals(res, len("YKQDIEAEGNYNFLEK"))


class Test_get_msa_num_sequences(unittest.TestCase):
    def test(self):
        res = get_msa_num_sequences("./tests/io_tests/1e7l_1_A.txt")
        self.assertEquals(res, 8)


class Test_get_msa_num_residues(unittest.TestCase):
    def test_no_gaps(self):
        res = get_msa_num_residues(
            "./tests/io_tests/1e7l_1_A.txt",
            exclude_gaps=False,
        )
        self.assertEquals(res, len("YKQDIEAEGNYNFLEK") * 8)

    def test_gaps(self):
        res = get_msa_num_residues(
            "./tests/io_tests/1e7l_1_A.txt",
            exclude_gaps=True,
        )
        self.assertEquals(res, len("YKQDIEAEGNYNFLEK") * 8 - 12)
