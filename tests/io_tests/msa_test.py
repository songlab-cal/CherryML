import unittest

from cherryml.io import get_msa_num_sites


class Test_get_msa_num_sites(unittest.TestCase):
    def test_l_infty_norm(self):
        res = get_msa_num_sites("./tests/io_tests/1e7l_1_A.txt")
        self.assertEquals(res, len("YKQDIEAEGNYNFLEK"))
