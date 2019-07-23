##
# \brief Tests mcmc estimation of copula paramters
from __future__ import print_function, division
import unittest
from scipy.stats.mstats import rankdata
# COPULA IMPORTS
from starvine.bvcopula.copula.gauss_copula import GaussCopula as gc
import numpy as np
import os
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)
tol = 4e-2


class TestMcmcFit(unittest.TestCase):
    def testMcmcCoplulaFit(self):
        print("--------------------- MCMC COPULA FIT TEST --------------------------")
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]

        # Rank transform the data
        u = rankdata(x) / (len(x) + 1)
        v = rankdata(y) / (len(y) + 1)

        # Fit t copula and gaussian copula
        thetag0 = [0.2]
        g_copula = gc()
        theta_g_fit_mle = g_copula.fitMLE(u, v, *thetag0, bounds=((-0.99, 0.99),))[0]
        aic_g_fit_mle = g_copula._AIC(u, v, 0, *theta_g_fit_mle)
        theta_g_fit_mcmc = g_copula.fitMcmc(u, v, *thetag0, bounds=((-0.99, 0.99),),
                                            ngen=500, nburn=200)[0]
        aic_g_fit_mcmc = g_copula._AIC(u, v, 0, *theta_g_fit_mcmc)
        print("Gaussian copula MLE paramter [rho]: " + str(theta_g_fit_mle) + " AIC =" + str(aic_g_fit_mle))
        print("Gaussian copula MCMC paramter [rho]: " + str(theta_g_fit_mcmc) + " AIC =" + str(aic_g_fit_mcmc))

        # check MLE and MCMC solution are the same in this case
        self.assertAlmostEqual(theta_g_fit_mle[0], theta_g_fit_mcmc[0], delta=tol)

        # check againt expected
        true_rho_ranked = 0.7387
        self.assertAlmostEqual(theta_g_fit_mcmc[0], true_rho_ranked, delta=tol)
