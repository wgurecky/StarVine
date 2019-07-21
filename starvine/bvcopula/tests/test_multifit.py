##
# \brief Test ability to determine best fit copula via AIC
from __future__ import print_function, division
from starvine.bvcopula.pc_base import PairCopula
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
        self.assertAlmostEqual(stockModel.copulaParams[1][0], 0.73874003, 4)

        # Test kendalls criterion on original data
        stockModel.copulaTournament(criterion='Kc', log=True)

        # When using kendalls criterion the predicted copula should be gumbel
        self.assertTrue(stockModel.copulaModel.name == "gumbel")
        self.assertTrue(stockModel.copulaParams[0] == "gumbel")
        self.assertTrue(stockModel.copulaModel.rotation == 0)

        # Rotate the data and test kendalls criteron again
        stockModelNegative = PairCopula(-1 * x, y)
        empTau = stockModelNegative.empKTau()[0]
        self.assertTrue(empTau < 0)

        # Test kendalls criterion on original data
        stockModelNegative.copulaTournament(criterion='Kc')

        # When using kendalls criterion the predicted copula should be gumbel
        self.assertTrue(stockModelNegative.copulaModel.name == "gumbel")
        self.assertTrue(stockModelNegative.copulaParams[0] == "gumbel")
        self.assertTrue(stockModelNegative.copulaModel.rotation == 1)

        # Rotate the data and test kendalls criteron again
        stockModelNegative = PairCopula(x, -1 * y)
        empTau = stockModelNegative.empKTau()[0]
        self.assertTrue(empTau < 0)

        # Test kendalls criterion on original data
        stockModelNegative.copulaTournament(criterion='Kc')

        # When using kendalls criterion the predicted copula should be gumbel
        self.assertTrue(stockModelNegative.copulaModel.name == "gumbel")
        self.assertTrue(stockModelNegative.copulaParams[0] == "gumbel")
        self.assertTrue(stockModelNegative.copulaModel.rotation == 3)
