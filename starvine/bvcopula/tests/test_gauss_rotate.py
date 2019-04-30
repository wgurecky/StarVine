##
# \brief Test copula rotations

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


class TestRotateGauss(unittest.TestCase):
    def testGaussRotate(self):
        np.random.seed(123)
        shapes = {0: 0.7777777, 1: -0.7777777, 2: 0.7777777, 3: -0.7777777}
        family = {'gauss': 0}
        for rotation, shapeParam in iteritems(shapes):
            gauss = Copula("gauss", 0)
            u, v = gauss.sample(10000, *(shapeParam,))
            g = sns.jointplot(u, v, stat_func=kendalltau)
            g.savefig("gauss_sample_pdf_" + str(rotation) + ".png")
            gauss.fittedParams = (shapeParam,)
            c_kTau = gauss.kTau()
            # compute rank corr coeff from resampled data
            gauss_model = PairCopula(u, v, family=family)
            gauss_model.copulaTournament()
            print(gauss_model.copulaParams)
            self.assertTrue("gauss" in gauss_model.copulaModel.name)
            # Ensure fitted shape parameter is same as original
            self.assertAlmostEqual(shapeParam, gauss_model.copulaParams[1][0], delta=0.2)
            # Ensure kTau is nearly the same from resampled data
            self.assertAlmostEqual(c_kTau, gauss_model.copulaModel.kTau(), delta=0.02)
            # fit to resampled data
            u_model, v_model = gauss_model.copulaModel.sample(10000)
            gauss_refit = PairCopula(u_model, v_model, family=family)
            gauss_refit.copulaTournament()
            u_resample, v_resample = gauss_refit.copulaModel.sample(2000)
            self.assertAlmostEqual(c_kTau, gauss_refit.copulaModel.kTau(), delta=0.05)
            self.assertAlmostEqual(shapeParam, gauss_refit.copulaParams[1][0], delta=0.2)
            # plot resampled data
            g_resample = sns.jointplot(u_resample, v_resample, stat_func=kendalltau)
            g_resample.savefig("gauss_resample_pdf_" + str(rotation) + ".png")
