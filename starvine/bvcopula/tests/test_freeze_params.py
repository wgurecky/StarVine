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


class TestGaussFrozen(unittest.TestCase):
    print("---------------------- COPULA FREEZE PARAMS TEST ---------------------")
    def testGaussFrozen(self):
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
        frzU, frzV = stockModel.copulaModel.sample(2000000)

        # Eval a model with specified params
        setU, setV = stockModel.copulaModel.sample(2000000, (0.73874003,))

        # Ensure both frozen model and specified param model produce same result
        frzModel = PairCopula(frzU, frzV)
        setModel = PairCopula(setU, setV)
        frzKtau, fp = frzModel.empKTau()
        setKtau, sp = setModel.empKTau()
        self.assertAlmostEqual(frzKtau, setKtau, places=3)
        self.assertAlmostEqual(fp, sp, places=3)

    def testFrankFrozen(self):
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        stockModel = PairCopula(x, y, family={'frank': 0, })

        # Try to fit all copula
        stockModel.copulaTournament(verbosity=0)

        # Eval the frozen model
        frzU, frzV = stockModel.copulaModel.sample(1000000)

        # Eval a model with specified params
        setU, setV = stockModel.copulaModel.sample(1000000, *stockModel.copulaParams[1])

        # Ensure both frozen model and specified param model produce same result
        frzModel = PairCopula(frzU, frzV)
        setModel = PairCopula(setU, setV)
        frzKtau, fp = frzModel.empKTau()
        setKtau, sp = setModel.empKTau()
        self.assertAlmostEqual(frzKtau, setKtau, places=3)
        self.assertAlmostEqual(fp, sp, places=3)
