#!/usr/bin/env python2

from __future__ import print_function, division
from starvine.uvar.uvmodels.uv_beta import UVBeta
import numpy as np
import unittest
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/testdata/"
tol = 0.1


class TestBetaFit(unittest.TestCase):
    """!
    @brief Check correctness of all fitting methods:
        - Maximum Likelihood
        - Method of Moments
        - Marcov Chain Monte Carlo
    """
    @classmethod
    def setUpClass(self):
        np.random.seed(123)
        self.true_model_params = np.array([2., 2.])
        self.model = UVBeta()
        # Obtain 1000 samples from beta PDF
        self.sampled_density_data = self.model.rvs(*self.true_model_params, size=1000)

    def testBetaFit(self):
        """!
        @brief Try all fitting methods.  Ensure all are in agreement.
        """
        print("---------------------- BETA FIT TEST ------------------------------")
        # init guess for parameters
        params0 = [1.8, 5.]
        tstData = self.sampled_density_data

        # ------------------------------------------------------------------------ #
        # Scipy's MLE estimate (fixed location and scale)
        mle_fitted_params = self.model.fit(tstData, *params0, floc=0, fscale=1)
        print("---- Scipy MLE params ----")
        print(mle_fitted_params)
        self.mle_fitted_params = np.array(mle_fitted_params[:-2])
        self.assertTrue(np.allclose(self.true_model_params, self.mle_fitted_params,
                                    atol=1e-4, rtol=tol))

        # ------------------------------------------------------------------------ #
        # Custom MLE estimate
        cmle_fitted_params = self.model.fitMLE(tstData, params0, bounds=((0, 50), (0, 50)))
        print("---- Custom MLE params ----")
        print(cmle_fitted_params)
        self.cmle_fitted_params = np.array(cmle_fitted_params)
        self.assertTrue(np.allclose(self.true_model_params, self.cmle_fitted_params,
                                    atol=1e-4, rtol=tol))

        # ------------------------------------------------------------------------ #
        # GMM estimate
        self.model.setupGMM(tstData, nMoM=4)
        gmm_params = self.model.internalGMM.fit(params0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
        print("---- GMM params ----")
        print(gmm_params.params)
        self.gmm_fitted_params = np.array(gmm_params.params)
        self.assertTrue(np.allclose(self.true_model_params, self.gmm_fitted_params,
                                    atol=1e-4, rtol=tol))

        # ------------------------------------------------------------------------ #
        # MCMC estimate
        # Define prior distributions for the paramers in the model
        def logPrior(theta):
            # Define ln(P(model))
            # Assume a and b are uncorrelated ln(P(a,b)) = ln(P(a))*ln(P(b))
            # Assume flat prior: ln(P(a)) == ln(1.0) == ln(P(b))
            a, b = theta
            if a <= 0 or b <= 0:
                # impossible condition.  Return ln(0.0)
                return -np.inf
            return 0.0  # ln(1.0) == 0.0
        self.model.setLogPrior(logPrior)
        # Set inital position of all walkers
        self.model.setupMCMC(20, params0, tstData)
        self.model.fitMCMC(1000)
        # Get results, discard first 500 samples
        samples = self.model.sampler.chain[:, 200:, :].reshape((-1, 2))
        mcmc_params = np.average(samples, axis=0)
        print("---- MCMC params ----")
        print("Averge: " + str(mcmc_params) + " +/-sigma :" + str(np.std(samples, axis=0)))
        self.mcmc_fitted_params = np.array(mcmc_params)

    @classmethod
    def tearDownClass(self):
        pass
