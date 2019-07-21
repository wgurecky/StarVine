##
# \brief Test copula rotations
from __future__ import print_function, division
from starvine.bvcopula.pc_base import PairCopula
from starvine.bvcopula.copula_factory import Copula
import unittest
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau
from six import iteritems
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestRotateFrankCopula(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(123)

    def testFrankRotate(self):
        expectedKtaus = {0: 0.602619667, 1: -0.602619667, 2: 0.602619667, 3: -0.602619667}
        shapeParam = 8.0
        family = {'gauss': 0,
                  'frank': 0,
                  'frank-90': 1,
                  'frank-180': 2,
                  'frank-270': 3,
                  }
        for rotation, expectedKtau in iteritems(expectedKtaus):
            frank = Copula("frank", rotation)
            u, v = frank.sample(10000, *(shapeParam,))
            g = sns.jointplot(u, v, stat_func=kendalltau)
            g.savefig("frank_sample_pdf_" + str(rotation) + ".png")
            frank.fittedParams = (shapeParam,)
            c_kTau = frank.kTau()
            # check kTau
            self.assertAlmostEqual(c_kTau, expectedKtau, delta=0.001)
            # compute rank corr coeff from resampled data
            frank_model = PairCopula(u, v, family=family)
            frank_model.copulaTournament()
            print(frank_model.copulaParams)
            self.assertTrue("frank" in frank_model.copulaModel.name)
            # Ensure refitted shape parameter is same as original
            self.assertAlmostEqual(shapeParam, frank_model.copulaParams[1][0], delta=0.2)
            # Ensure kTau is nearly the same from resampled data
            self.assertAlmostEqual(c_kTau, frank_model.copulaModel.kTau(), delta=0.02)
            # fit to resampled data
            u_model, v_model = frank_model.copulaModel.sample(10000)
            frank_refit = PairCopula(u_model, v_model, family=family)
            frank_refit.copulaTournament()
            u_resample, v_resample = frank_refit.copulaModel.sample(1000)
            self.assertAlmostEqual(c_kTau, frank_refit.copulaModel.kTau(), delta=0.05)
            # plot resampled data
            g_resample = sns.jointplot(u_resample, v_resample, stat_func=kendalltau)
            g_resample.savefig("frank_resample_pdf_" + str(rotation) + ".png")
