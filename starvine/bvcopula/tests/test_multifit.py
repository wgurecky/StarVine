##
# \brief Test ability to determine best fit copula via AIC
from __future__ import print_function, division
from pc_base import PairCopula
import unittest
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
        stockModel = PairCopula(x, y)

        empTau = stockModel.empKTau()    # kendall's tau corr coeff
        empSRho = stockModel.empSRho()  # spearmans corr coeff
        empPRho = stockModel.empPRho()  # pearson's corr coeff
        print("Emprical kTau, spearmanr, pearsonr: " +
              str(empTau[0]) + ", " + str(empSRho[0]) + ", " + str(empPRho[0]))

        # Try to fit all copula
        stockModel.copulaTournament()

        # Ensure that the gaussian copula was chosen as the best fit
        self.assertTrue(stockModel.copulaModel.name == "gauss")
        self.assertTrue(stockModel.copulaParams[0] == "gauss")

        # Check gaussian copula parameters for correctness
        self.assertAlmostEqual(stockModel.copulaParams[1], 0.73874003, 4)
