##
# \brief Test copula CDF's to ensure integral on [0,1]^2 is 1
from __future__ import print_function, division
# COPULA IMPORTS
from starvine.bvcopula.copula.t_copula import StudentTCopula
from starvine.bvcopula.copula.gauss_copula import GaussCopula
from starvine.bvcopula.copula.frank_copula import FrankCopula
from starvine.bvcopula.copula.gumbel_copula import GumbelCopula
from starvine.bvcopula.copula.clayton_copula import ClaytonCopula
from starvine.bvcopula.copula.indep_copula import IndepCopula
#
import unittest
import numpy as np
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestCopulaCDF(unittest.TestCase):
    def testCopulaCDF(self):
        # check all CDFs == 1.0 at U, V = [1.0, 1.0]
        print("------------------- COPULA CDF INTEGRAL TEST ----------------------")

    def testTCopulaCDF(self):
        t_copula = StudentTCopula()
        u, v = np.ones(1) - 1e-12, np.ones(1) - 1e-12
        cdf_max = t_copula.cdf(u, v, *[0.7, 10])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testGaussCopulaCDF(self):
        gauss_copula = GaussCopula()
        u, v = np.ones(1) - 1e-9, np.ones(1) - 1e-9
        cdf_max = gauss_copula.cdf(u, v, *[0.7])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testFrankCopulaCDF(self):
        frank_copula = FrankCopula()
        u, v = np.ones(1), np.ones(1)
        cdf_max = frank_copula.cdf(u, v, *[2.7])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testGumbelCopulaCDF(self):
        gumbel_copula = GumbelCopula()
        u, v = np.ones(1), np.ones(1)
        cdf_max = gumbel_copula.cdf(u, v, *[2.7])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testClaytonCopulaCDF(self):
        clayton_copula = ClaytonCopula()
        u, v = np.ones(1), np.ones(1)
        cdf_max = clayton_copula.cdf(u, v, *[2.7])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testClayton90CopulaCDF(self):
        clayton_copula = ClaytonCopula(1)
        u, v = np.ones(1), np.ones(1)
        cdf_max = clayton_copula.cdf(u, v, *[2.7])
        self.assertAlmostEqual(cdf_max[0], 1.0)

    def testIndepCopulaCDF(self):
        indep_copula = IndepCopula()
        u, v = np.ones(1), np.ones(1)
        cdf_max = indep_copula.cdf(u, v)
        self.assertAlmostEqual(cdf_max[0], 1.0)
