##
# \brief Test copula mle fit with weighted samples
from __future__ import print_function, division
import unittest
import numpy as np
from scipy.stats import norm
import seaborn as sns
from six import iteritems
import os
import pandas as pd
# starvine imports
from starvine.bvcopula.pc_base import PairCopula
from starvine.bvcopula.copula_factory import Copula
from starvine.mvar.mv_plot import matrixPairPlot
#
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestWeightedReg(unittest.TestCase):
    def testWgtCopula(self):
        """!
        @brief Test ability to construct copula
        given samples with unequal weights.
        Compose two bivariate gauss dists, one with
        positive and one with negative depencence.
        Sample from dists.
        Assign large sample weights to positive gauss
        and low sample weights to neg gauss.
        Combine weighted samples into a single "X" shaped distribution.
        Refit weighted samples and ensure positive depencence
        """
        np.random.seed(123)
        # construct gaussian margins; mu={0, 0}, sd={1.0, 2}
        # marg1 = Uvm("gauss")(1e-3, 1.)
        marg1 = norm(loc=1e-3, scale=1.0)
        # marg2 = Uvm("gauss")(1e-3, 2.)
        marg2 = norm(loc=1e-3, scale=2.0)

        # construct gaussian copula positive dep
        cop1 = Copula("gauss")
        cop1.fittedParams = [0.7]

        # construct gaussian copula neg dep
        cop2 = Copula("gauss")
        cop2.fittedParams = [-0.7]

        # draw 4000 samples from each model
        n = 4000
        x1, y1 = cop1.sampleScale(marg1, marg2, n)
        x2, y2 = cop2.sampleScale(marg1, marg2, n)

        # assign weights to each gauss sample group
        cop1_wgts = np.ones(n) * 0.95
        cop2_wgts = np.ones(n) * 0.05

        # combine both gauss models into dbl gauss model
        x = np.append(x1, x2)
        y = np.append(y1, y2)
        wgts = np.append(cop1_wgts, cop2_wgts)

        # plot
        data = pd.DataFrame([x, y]).T
        matrixPairPlot(data, weights=wgts, savefig='x_gauss_original.png')

        # fit copula to weighted data
        copModel = PairCopula(x, y, wgts)
        copModel.copulaTournament()

        # verify that a positive dep copula was produced with a
        # dep parameter of slightly less than 0.7
        x_wt, y_wt = copModel.copulaModel.sampleScale(marg1, marg2, n)
        self.assertTrue(copModel.copulaModel.kTau() > 0.)
        self.assertTrue((copModel.copulaModel.fittedParams[0] > 0.)
                        & (copModel.copulaModel.fittedParams[0] < 0.7))

        # plot
        data = pd.DataFrame([x_wt, y_wt]).T
        matrixPairPlot(data, savefig='x_gauss_weighted_fit.png')

    def testWgtResampledCopula(self):
        """!
        @brief Test ability to construct copula
        given samples with unequal weights using a resampling strat
        """
        np.random.seed(123)
        # construct gaussian margins; mu={0, 0}, sd={1.0, 2}
        # marg1 = Uvm("gauss")(1e-3, 1.)
        marg1 = norm(loc=1e-3, scale=1.0)
        # marg2 = Uvm("gauss")(1e-3, 2.)
        marg2 = norm(loc=1e-3, scale=2.0)

        # construct gaussian copula positive dep
        cop1 = Copula("gauss")
        cop1.fittedParams = [0.7]

        # construct gaussian copula neg dep
        cop2 = Copula("gauss")
        cop2.fittedParams = [-0.7]

        # draw 1000 samples from each model
        n = 1000
        x1, y1 = cop1.sampleScale(marg1, marg2, n)
        x2, y2 = cop2.sampleScale(marg1, marg2, n)

        # assign weights to each gauss sample group
        cop1_wgts = np.ones(n) * 0.95
        cop2_wgts = np.ones(n) * 0.05

        # combine both gauss models into dbl gauss model
        x = np.append(x1, x2)
        y = np.append(y1, y2)
        wgts = np.append(cop1_wgts, cop2_wgts)

        # fit copula to weighted data
        copModel = PairCopula(x, y, wgts, resample=10)
        copModel.copulaTournament()

        resampled_data = pd.DataFrame([copModel.x, copModel.y]).T
        matrixPairPlot(resampled_data, savefig='x_gauss_resampled.png')

        # verify that a positive dep copula was produced with a
        # dep parameter of slightly less than 0.7
        x_wt, y_wt = copModel.copulaModel.sampleScale(marg1, marg2, n)
        self.assertTrue(copModel.copulaModel.kTau() > 0.)
        self.assertTrue((copModel.copulaModel.fittedParams[0] > 0.)
                        & (copModel.copulaModel.fittedParams[0] < 0.7))

        # plot
        data = pd.DataFrame([x_wt, y_wt]).T
        matrixPairPlot(data, savefig='x_gauss_resampled_fit.png')
