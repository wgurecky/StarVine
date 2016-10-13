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
        print("----------------- COPULA FREEZE PARAMS TEST --------------------")
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        stockModel = PairCopula(x, y)

        # Try to fit all copula
        stockModel.copulaTournament(verbosity=0)

        # Ensure that the gaussian copula was chosen as the best fit
        self.assertTrue(stockModel.copulaModel.name == "gauss")
        self.assertTrue(stockModel.copulaParams[0] == "gauss")

        # Check gaussian copula parameters for correctness
        self.assertAlmostEqual(stockModel.copulaParams[1], 0.73874003, 4)

        # Eval the frozen model
        resampledU, resampledV = stockModel.copulaModel.sample(1000)
