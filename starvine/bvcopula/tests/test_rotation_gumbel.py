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

    def testGumbelRotate(self):
        expectedKtaus = {0: 0.87499, 1: -0.87499, 2: 0.87499, 3: -0.87499}
        shapeParam = 8.0
        family = {'gumbel': 0,
                 'gumbel-90': 1,
                 'gumbel-180': 2,
                 'gumbel-270': 3}
        for rotation, expectedKtau in iteritems(expectedKtaus):
            gumbel = Copula("gumbel", rotation)
            u, v = gumbel.sample(40000, *(shapeParam,))
            g = sns.jointplot(u, v)
            g.savefig("gumbel_sample_pdf_" + str(rotation) + ".png")
            gumbel.fittedParams = (shapeParam,)
            c_kTau = gumbel.kTau()
            # check kTau
            self.assertAlmostEqual(c_kTau, expectedKtau, delta=0.001)
            # compute rank corr coeff from resampled data
            gumbel_model = PairCopula(u, v, family=family)
            gumbel_model.copulaTournament()
            print(gumbel_model.copulaParams)
            self.assertTrue("gumbel" in gumbel_model.copulaModel.name)
            # Ensure fitted shape parameter is same as original
            self.assertAlmostEqual(shapeParam, gumbel_model.copulaParams[1][0], delta=0.2)
            self.assertEqual(rotation, gumbel_model.copulaParams[3])
            # Ensure kTau is nearly the same from resampled data
            self.assertAlmostEqual(c_kTau, gumbel_model.copulaModel.kTau(), delta=0.02)
            # fit to resampled data
            u_model, v_model = gumbel_model.copulaModel.sample(40000)
            gumbel_refit = PairCopula(u_model, v_model, family=family)
            gumbel_refit.copulaTournament()
            u_resample, v_resample = gumbel_refit.copulaModel.sample(4000)
            self.assertAlmostEqual(c_kTau, gumbel_refit.copulaModel.kTau(), delta=0.05)
            # plot resampled data
            g_resample = sns.jointplot(u_resample, v_resample)
            g_resample.savefig("gumbel_resample_pdf_" + str(rotation) + ".png")
