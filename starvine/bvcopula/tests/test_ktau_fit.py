##
# \brief Test ability to fit copula by specified Ktau
from __future__ import print_function, division
from starvine.bvcopula.pc_base import PairCopula
from starvine.bvcopula.copula_factory import Copula
import unittest
import numpy as np
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestKtauFit(unittest.TestCase):
    def testKtauFitGauss(self):
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        stockModel = PairCopula(x, y)

        # Try to fit all copula
        stockModel.copulaTournament()

        # Check gaussian copula parameters for correctness
        self.assertAlmostEqual(stockModel.copulaParams[1], 0.73874003, 4)

        # Obtain Ktau
        kTauFitted = stockModel.copulaModel.kTau()

        # Use fitted kTau to specify a new copula
        stockModelRefit = Copula("gauss")

        # Fit gauss copula param given kTau
        fittedParam = stockModelRefit.fitKtau(kTauFitted)[0]
        print("Fitted Ktau = %f" % kTauFitted)
        print("Expected Theta = %f" % 0.738740)
        print("Computed Theta = %f" % fittedParam)

        # check result
        self.assertAlmostEqual(0.73874003, fittedParam, delta=0.01)



