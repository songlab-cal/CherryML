import unittest

import numpy as np

from cherryml.evaluation import l_infty_norm, mre, rmse


class TestLInftyNorm(unittest.TestCase):
    def test_l_infty_norm(self):
        Q_1 = np.array([[-1.0, 1.0], [2.0, -2.0]])
        Q_2 = np.array([[-2.0, 2.0], [3.0, -3.0]])
        np.testing.assert_almost_equal(l_infty_norm(Q_1, Q_2), np.log(2))
        np.testing.assert_almost_equal(l_infty_norm(Q_2, Q_1), np.log(2))
        np.testing.assert_almost_equal(l_infty_norm(Q_1, Q_1), 0.0)


class TestRMSE(unittest.TestCase):
    def test_l_infty_norm(self):
        Q_1 = np.array([[-1.0, 1.0], [2.0, -2.0]])
        Q_2 = np.array([[-2.0, 2.0], [3.0, -3.0]])
        np.testing.assert_almost_equal(rmse(Q_1, Q_2), 0.5678269844632537)
        np.testing.assert_almost_equal(rmse(Q_2, Q_1), 0.5678269844632537)
        np.testing.assert_almost_equal(rmse(Q_1, Q_1), 0.0)


class TestMRE(unittest.TestCase):
    def test_mre(self):
        Q_1 = np.array([[-2.0, 2.0], [3.0, -3.0]])
        Q_2 = np.array([[-3.0, 3.0], [5.0, -5.0]])
        np.testing.assert_almost_equal(mre(Q_1, Q_2), 2 / 3)
        np.testing.assert_almost_equal(mre(Q_2, Q_1), 2 / 3)
        np.testing.assert_almost_equal(mre(Q_1, Q_1), 0.0)
