##
# \brief Tests ability to fit and sample a mixture of 2 copula
from __future__ import print_function, division
import unittest
from scipy.stats.mstats import rankdata
# COPULA IMPORTS
from starvine.bvcopula.copula.mixture_copula import MixtureCopula as mc
import numpy as np
import matplotlib.pyplot as plt
import os
pwd_ = os.path.dirname(os.path.abspath(__file__))
from starvine.bvcopula.copula.gumbel_copula import GumbelCopula
dataDir = pwd_ + "/data/"
np.random.seed(123)
tol = 4e-2


class TestMixtureCopula(unittest.TestCase):
    def setUp(self):
        self._mix_copula = mc(GumbelCopula(1), 0.5,
                              GumbelCopula(2), 0.5)

    def testMixCoplulaPdf(self):
        u = np.linspace(1.0e-8, 1.0-1e-8, 50)
        v = np.linspace(1.0e-8, 1.0-1e-8, 50)
        c_pdf = self._mix_copula.pdf(u, v)
        self.assertTrue(np.all(c_pdf >= 0))

    def testMixCoplulaCdf(self):
        u = np.linspace(1.0e-8, 1.0-1e-8, 50)
        v = np.linspace(1.0e-8, 1.0-1e-8, 50)
        c_pdf = self._mix_copula.cdf(u, v)
        self.assertTrue(np.all(c_pdf >= 0))
        self.assertTrue(np.all(c_pdf <= 1))

    def testMixCoplulaSample(self):
        pass
