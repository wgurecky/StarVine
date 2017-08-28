##
# \brief Test ability to compute kendall's function
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

        # Compute kendall's function
        t_in = np.linspace(0.01, 0.99, 20)
        k_c = stockModelRefit.kC(t_in)
        print("Gauss Kendall's function")
        print(t_in)
        print(k_c)

        # And specify gumbel model
        stockModelRefit = Copula("gumbel")
        fittedParam = stockModelRefit.fitKtau(kTauFitted)[0]
        k_c_gumbel = stockModelRefit.kC(t_in)
        print("Gumbel Kendall's function")
        print(t_in)
        print(k_c_gumbel)
        print("Fitted Gumbel Param= %f" % fittedParam)
        # expected gubel param for ktau=0.9, 10.0
        # expected gubel param for ktau=0.8, 5.0
        # expected gubel param for ktau=0.7, 3.33

        # check that k_c is monotonic
        self.assertTrue(k_c[0] < k_c[1])
        self.assertTrue(k_c[-2] < k_c[-1])

        # Compute emperical kendall's function
        t_emp, kc_emp = stockModel.empKc()

        # plot
        try:
            import pylab as pl
            pl.figure()
            pl.plot(t_in, t_in - k_c, label="gauss")
            pl.plot(t_in, t_in - k_c_gumbel, label="gumbel")
            pl.plot(t_emp, t_emp - kc_emp, label="emperical")
            pl.xlabel("t")
            pl.ylabel("t - Kc(t)")
            pl.legend()
            pl.savefig("ktau_function_plot.png")
            pl.close()
        except:
            pass
