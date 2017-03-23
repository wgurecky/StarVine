##
# \brief Test copula mle fit with weighted samples
import unittest
import numpy as np
import seaborn as sns
from six import iteritems
import os
# starvine imports
from __future__ import print_function, division
from starvine.bvcopula.pc_base import PairCopula
from starvine.bvcopula.copula_factory import Copula
from starvine.uvar.uvmodel_factory import Uvm
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
        # construct gaussian margins; mu={0, 0}, sd={1.0, 2}
        marg1 = Uvm("gauss")([0., 1.0])
        marg2 = Uvm("gauss")([0., 2.0])

        # construct gaussian copula positive dep
        cop1 = Copula("gauss")
        cop1.fittedParams([0.7])

        # construct gaussian copula neg dep
        cop2 = Copula("gauss")
        cop2.fittedParams([-0.7])

        # draw 1000 samples from each model
        n = 1000
        rvs1 = marg1.rvs(n)
        rvs2 = marg2.rvs(n)
        x1, y1 = cop1.sampleScale(rvs1, rvs2, marg1.cdf, marg2.cdf)
        x2, y2 = cop2.sampleScale(rvs1, rvs2, marg1.cdf, marg2.cdf)

        # assign weights to each gauss sample group
        cop1_wgts = np.ones(n) * 0.9
        cop2_wgts = np.ones(n) * 0.1

        # combine both gauss models and plot
        x = np.append(x1, x2)
        y = np.append(y1, y2)
        wgts = np.append(cop1_wgts, cop2_wgts)

        # fit copula to weighted data
        copModel = PairCopula(x1, y1, wgts)
        copModel.copulaTournament()

        # verify that a positive dep copula was produced with a
        # dep parameter of slightly less than 0.7
        pass


    def testWgtMargins(self):
        """!
        @brief Test the ability to construct a marginal PDF
        given samples with unequal weights.
        """
        pass
