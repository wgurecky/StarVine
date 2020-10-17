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


class TestRotateArchimedeanCopula(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(123)

    def testClaytonRotate(self):
        expectedKtaus = {0: 0.7777777, 1: -0.7777777, 2: 0.7777777, 3: -0.7777777}
        shapeParam = 7.0
        family = {'gauss': 0,
                  'clayton': 0,
                  'clayton-90': 1,
                  'clayton-180': 2,
                  'clayton-270': 3,
                  'gumbel': 0,
                  'gumbel-90': 1,
                  'gumbel-180': 2,
                  'gumbel-270': 3,
                  }
        for rotation, expectedKtau in iteritems(expectedKtaus):
            clayton = Copula("clayton", rotation)
            u, v = clayton.sample(40000, *(shapeParam,))
            g = sns.jointplot(u, v)
            g.savefig("clayton_sample_pdf_" + str(rotation) + ".png")
            clayton.fittedParams = (shapeParam,)
            c_kTau = clayton.kTau()
            # check kTau
            self.assertAlmostEqual(c_kTau, expectedKtau, delta=0.001)
            # compute rank corr coeff from resampled data
            clayton_model = PairCopula(u, v, family=family)
            clayton_model.copulaTournament()
            print(clayton_model.copulaParams)
            self.assertTrue("clayton" in clayton_model.copulaModel.name)
            # Ensure fitted shape parameter is same as original
            self.assertAlmostEqual(shapeParam, clayton_model.copulaParams[1][0], delta=0.2)
            self.assertEqual(rotation, clayton_model.copulaParams[3])
            # Ensure kTau is nearly the same from resampled data
            self.assertAlmostEqual(c_kTau, clayton_model.copulaModel.kTau(), delta=0.02)
            # fit to resampled data
            u_model, v_model = clayton_model.copulaModel.sample(10000)
            clayton_refit = PairCopula(u_model, v_model, family=family)
            clayton_refit.copulaTournament()
            u_resample, v_resample = clayton_refit.copulaModel.sample(1000)
            self.assertAlmostEqual(c_kTau, clayton_refit.copulaModel.kTau(), delta=0.05)
            # plot resampled data
            g_resample = sns.jointplot(u_resample, v_resample)
            g_resample.savefig("clayton_resample_pdf_" + str(rotation) + ".png")
