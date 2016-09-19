##
# \brief Test ability to determine best fit copula via AIC
from __future__ import print_function, division
import unittest
import bv_base as bvb
import numpy as np
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestBivariateBase(unittest.TestCase):
    def testBivariateBase(self):
        print("--------------------- MULTI COPULA FIT TEST --------------------------")
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        stocks = bvb.BVbase(x, y)

        empTau = stocks.empKTau()    # kendall's tau corr coeff
        empSRho = stocks.empSRho()  # spearmans corr coeff
        empPRho = stocks.empPRho()  # pearson's corr coeff
        print("Emprical kTau, spearmanr, pearsonr: " +
              str(empTau[0]) + ", " + str(empSRho[0]) + ", " + str(empPRho[0]))

        # Try to fit all copula
        stocks.copulaTournament()

        # Ensure that the gaussian copula was chosen as the best fit
        self.assertTrue(stocks.copulaModel[0].name == "gauss")
        self.assertTrue(stocks.copulaModel[1][0] == "gauss")

        # Check gaussian copula parameters for correctness
        self.assertAlmostEqual(stocks.copulaModel[1][1][0], 0.73874003, 4)
