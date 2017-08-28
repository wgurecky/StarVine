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


class TestGaussFrozen(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(123)

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
        frzU, frzV = stockModel.copulaModel.sample(40000)

        # Eval a model with specified params
        setU, setV = stockModel.copulaModel.sample(40000, (0.73874003,))

        # Ensure both frozen model and specified param model produce same result
        frzModel = PairCopula(frzU, frzV)
        setModel = PairCopula(setU, setV)
        frzKtau, fp = frzModel.empKTau()
        setKtau, sp = setModel.empKTau()
        self.assertAlmostEqual(frzKtau, setKtau, delta=0.02)
        self.assertAlmostEqual(fp, sp, delta=0.02)

        # Eval a model with different specified params
        setU2, setV2 = stockModel.copulaModel.sample(20000, (0.3,))
        setModel2 = PairCopula(setU2, setV2)
        setKtau2, sp2 = setModel2.empKTau()
        self.assertTrue(setKtau2 != setKtau)
        self.assertTrue(abs(setKtau2 - setKtau) > 0.2)


    def testFrankFrozen(self):
        np.random.seed(123)
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        stockModel = PairCopula(x, y, family={'frank': 0, })

        # Try to fit all copula
        stockModel.copulaTournament(verbosity=0)

        # Eval the frozen model
        frzU, frzV = stockModel.copulaModel.sample(40000)

        # Eval a model with specified params
        setU, setV = stockModel.copulaModel.sample(40000, *stockModel.copulaParams[1])

        # Ensure both frozen model and specified param model produce same result
        frzModel = PairCopula(frzU, frzV)
        setModel = PairCopula(setU, setV)
        frzKtau, fp = frzModel.empKTau()
        setKtau, sp = setModel.empKTau()
        self.assertAlmostEqual(frzKtau, setKtau, delta=0.02)
        self.assertAlmostEqual(fp, sp, delta=0.02)
