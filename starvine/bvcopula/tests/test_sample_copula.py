from __future__ import print_function, division
import unittest
from starvine.bvcopula.pc_base import PairCopula
from starvine.bvcopula.copula_factory import Copula
import numpy as np


class TestSampleCopula(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(123)

    def testClaytonSample(self):
        # 0 deg
        clayton00 = Copula("clayton", 0)
        u00, v00 = clayton00.sample(10000, *(0.5,))
        clayton00.fittedParams = (0.5,)
        c00_kTau = clayton00.kTau()
        # check kTau
        self.assertAlmostEqual(c00_kTau, 0.2)
        # compute rank corr coeff from resampled data
        clayton00_model = PairCopula(u00, v00)
        clayton00_model.copulaTournament()
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c00_kTau, clayton00_model.copulaModel.kTau(), delta=0.02)

        # 90 deg
        clayton90 = Copula("clayton", 1)
        u90, v90 = clayton90.sample(10000, *(0.5,))
        clayton90.fittedParams = (0.5,)
        c90_kTau = clayton90.kTau()
        self.assertAlmostEqual(c90_kTau, -0.2)
        # compute rank corr coeff from resampled data
        clayton90_model = PairCopula(u90, v90, family={"clayton": 1})
        clayton90_model.copulaTournament()
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton90_model.copulaModel.name)
        self.assertTrue(1 == clayton90_model.copulaModel.rotation)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c90_kTau, clayton90_model.copulaModel.kTau(), delta=0.02)

        # 180 deg
        clayton180 = Copula("clayton", 2)
        u180, v180 = clayton180.sample(1000, *(0.5,))
        # 270 deg
        clayton270 = Copula("clayton", 3)
        u270, v270 = clayton270.sample(1000, *(0.5,))

    def testGumbelSample(self):
        # TODO: TEST FAILS FOR SMALL GUMBEL PARAMETER VALUES
        # 0 deg
        gumbel00 = Copula("gumbel", 0)
        u00, v00 = gumbel00.sample(10000, *(8.0,))
        gumbel00.fittedParams = (8.0,)
        c00_kTau = gumbel00.kTau()
        # check kTau
        self.assertAlmostEqual(c00_kTau, 0.8749999999999, 6)
        # compute rank corr coeff from resampled data
        gumbel00_model = PairCopula(u00, v00, family={"gumbel": 0})
        gumbel00_model.copulaTournament()
        print(gumbel00_model.copulaParams)
        # check that gumbel copula won since
        # we seeded with samples from a gumbel df
        self.assertTrue("gumbel" in gumbel00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c00_kTau, gumbel00_model.copulaModel.kTau(), delta=0.04)

        # 90 deg
        gumbel90 = Copula("gumbel", 1)
        u90, v90 = gumbel90.sample(10000, *(8.0,))
        gumbel90.fittedParams = (8.0,)
        c90_kTau = gumbel90.kTau()
        self.assertAlmostEqual(c90_kTau, -0.87499999999, 6)
        # compute rank corr coeff from resampled data
        gumbel90_model = PairCopula(u90, v90, family={"gumbel": 1})
        gumbel90_model.copulaTournament()
        print(gumbel90_model.copulaParams)
        # check that gumbel copula won since
        # we seeded with samples from a gumbel df
        self.assertTrue("gumbel" in gumbel90_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c90_kTau, gumbel90_model.copulaModel.kTau(), delta=0.04)

    def testFrankSample(self):
        # 0 deg
        frank00 = Copula("frank", 0)
        u00, v00 = frank00.sample(30000, *(8.0,))
        frank00.fittedParams = (8.0,)
        c00_kTau = frank00.kTau()
        # check kTau
        print(c00_kTau)
        self.assertAlmostEqual(c00_kTau, 0.602619667, 6)
        # compute rank corr coeff from resampled data
        frank00_model = PairCopula(u00, v00, family={"frank": 0})
        frank00_model.copulaTournament()
        print(frank00_model.copulaParams)
        # check that frank copula won since
        # we seeded with samples from a frank df
        self.assertTrue("frank" in frank00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c00_kTau, frank00_model.copulaModel.kTau(), delta=0.02)

        # 90 deg
        frank90 = Copula("frank", 1)
        u90, v90 = frank90.sample(30000, *(8.0,))
        frank90.fittedParams = (8.0,)
        c90_kTau = frank90.kTau()
        self.assertAlmostEqual(c90_kTau, -0.602619667, 6)
        # compute rank corr coeff from resampled data
        frank90_model = PairCopula(u90, v90, family={"frank": 1})
        frank90_model.copulaTournament()
        print(frank90_model.copulaParams)
        # check that frank copula won since
        # we seeded with samples from a frank df
        self.assertTrue("frank" in frank90_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        self.assertAlmostEqual(c90_kTau, frank90_model.copulaModel.kTau(), delta=0.02)

    def testGaussSample(self):
        pass

    def testTSample(self):
        pass

