##
# \brief Test copula rotations
from __future__ import print_function, division
from pc_base import PairCopula
from copula_factory import Copula
import unittest
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)


class TestRotateCopula(unittest.TestCase):
    def testGumbelRotate(self):
        # 0 deg
        gumbel00 = Copula("gumbel", 0)
        u00, v00 = gumbel00.sample(10000, *(8.0,))
        g00 = sns.jointplot(u00, v00, stat_func=kendalltau)
        g00.savefig("gumbel_sample_pdf_00.png")
        gumbel00.fittedParams = (8.0,)
        c00_kTau = gumbel00.kTau()
        # check kTau
        self.assertAlmostEqual(c00_kTau, 0.8749999999999, 6)
        # compute rank corr coeff from resampled data
        gumbel00_model = PairCopula(u00, v00)
        gumbel00_model.copulaTournament()
        print(gumbel00_model.copulaParams)
        # check that gumbel copula won since
        # we seeded with samples from a gumbel df
        self.assertTrue("gumbel" in gumbel00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c00_kTau - gumbel00_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.01)
        self.assertAlmostEqual(c00_kTau, gumbel00_model.copulaModel.kTau(), delta=0.01)
        # fit to resampled data
        u00_model, v00_model = gumbel00_model.copulaModel.sample(10000)
        gumbel00_refit = PairCopula(u00_model, v00_model)
        gumbel00_refit.copulaTournament()
        u00_resample, v00_resample = gumbel00_refit.copulaModel.sample(10000)
        g00_resample = sns.jointplot(u00_resample, v00_resample, stat_func=kendalltau)
        g00_resample.savefig("gumbel_resample_pdf_00.png")

        # 90 deg
        gumbel90 = Copula("gumbel", 1)
        u90, v90 = gumbel90.sample(10000, *(8.0,))
        g90 = sns.jointplot(u90, v90, stat_func=kendalltau)
        g90.savefig("gumbel_sample_pdf_90.png")
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
        kTauDelta = c90_kTau - gumbel90_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c90_kTau, gumbel90_model.copulaModel.kTau(), delta=0.02)
        # fit to resampled data
        u90_model, v90_model = gumbel90_model.copulaModel.sample(10000)
        gumbel90_refit = PairCopula(u90_model, v90_model)
        gumbel90_refit.copulaTournament()
        u90_resample, v90_resample = gumbel90_refit.copulaModel.sample(10000)
        g90_resample = sns.jointplot(u90_resample, v90_resample, stat_func=kendalltau)
        g90_resample.savefig("gumbel_resample_pdf_90.png")

        # 180 deg
        gumbel180 = Copula("gumbel", 2)
        u180, v180 = gumbel180.sample(10000, *(8.0,))
        g180 = sns.jointplot(u180, v180, stat_func=kendalltau)
        g180.savefig("gumbel_sample_pdf_180.png")
        gumbel180.fittedParams = (8.0,)
        c180_kTau = gumbel180.kTau()
        self.assertAlmostEqual(c00_kTau, 0.8749999999999, 6)
        # compute rank corr coeff from resampled data
        gumbel180_model = PairCopula(u180, v180, family={"gumbel": 2})
        gumbel180_model.copulaTournament()
        print(gumbel180_model.copulaParams)
        # check that gumbel copula won since
        # we seeded with samples from a gumbel df
        self.assertTrue("gumbel" in gumbel180_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c180_kTau - gumbel180_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c180_kTau, gumbel180_model.copulaModel.kTau(), delta=0.02)
        u180_model, v180_model = gumbel180_model.copulaModel.sample(10000)
        gumbel180_refit = PairCopula(u180_model, v180_model)
        gumbel180_refit.copulaTournament()
        u180_resample, v180_resample = gumbel180_refit.copulaModel.sample(10000)
        g180_resample = sns.jointplot(u180_resample, v180_resample, stat_func=kendalltau)
        g180_resample.savefig("gumbel_resample_pdf_180.png")

        # 270 deg
        gumbel270 = Copula("gumbel", 3)
        u270, v270 = gumbel270.sample(10000, *(8.0,))
        g270 = sns.jointplot(u270, v270, stat_func=kendalltau)
        g270.savefig("gumbel_sample_pdf_270.png")
        gumbel270.fittedParams = (8.0,)
        c270_kTau = gumbel270.kTau()
        self.assertAlmostEqual(c270_kTau, -0.87499999999, 6)
        # compute rank corr coeff from resampled data
        gumbel270_model = PairCopula(u270, v270, family={"gumbel": 3})
        gumbel270_model.copulaTournament()
        print(gumbel270_model.copulaParams)
        # check that gumbel copula won since
        # we seeded with samples from a gumbel df
        self.assertTrue("gumbel" in gumbel270_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c270_kTau - gumbel270_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c270_kTau, gumbel270_model.copulaModel.kTau(), delta=0.02)
        u270_model, v270_model = gumbel270_model.copulaModel.sample(10000)
        gumbel270_refit = PairCopula(u270_model, v270_model)
        gumbel270_refit.copulaTournament()
        u270_resample, v270_resample = gumbel270_refit.copulaModel.sample(10000)
        g270_resample = sns.jointplot(u270_resample, v270_resample, stat_func=kendalltau)
        g270_resample.savefig("gumbel_resample_pdf_270.png")

    def testFrankRotate(self):
        # 0 deg
        frank00 = Copula("frank", 0)
        u00, v00 = frank00.sample(10000, *(8.0,))
        g00 = sns.jointplot(u00, v00, stat_func=kendalltau)
        g00.savefig("frank_sample_pdf_00.png")
        frank00.fittedParams = (8.0,)
        c00_kTau = frank00.kTau()
        # check kTau
        self.assertAlmostEqual(c00_kTau, 0.602619667, 6)
        # compute rank corr coeff from resampled data
        frank00_model = PairCopula(u00, v00, family={"frank": 0})
        frank00_model.copulaTournament()
        print(frank00_model.copulaParams)
        # check that frank copula won since
        # we seeded with samples from a frank df
        self.assertTrue("frank" in frank00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c00_kTau - frank00_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.01)
        self.assertAlmostEqual(c00_kTau, frank00_model.copulaModel.kTau(), delta=0.01)

        # 90 deg
        frank90 = Copula("frank", 1)
        u90, v90 = frank90.sample(10000, *(8.0,))
        g90 = sns.jointplot(u90, v90, stat_func=kendalltau)
        g90.savefig("frank_sample_pdf_90.png")
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
        kTauDelta = c90_kTau - frank90_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c90_kTau, frank90_model.copulaModel.kTau(), delta=0.02)

        # 180 deg
        frank180 = Copula("frank", 2)
        u180, v180 = frank180.sample(10000, *(8.0,))
        g180 = sns.jointplot(u180, v180, stat_func=kendalltau)
        g180.savefig("frank_sample_pdf_180.png")
        frank180.fittedParams = (8.0,)
        c180_kTau = frank180.kTau()
        self.assertAlmostEqual(c00_kTau, 0.602619667, 6)
        # compute rank corr coeff from resampled data
        frank180_model = PairCopula(u180, v180, family={"frank": 2})
        frank180_model.copulaTournament()
        print(frank180_model.copulaParams)
        # check that frank copula won since
        # we seeded with samples from a frank df
        self.assertTrue("frank" in frank180_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c180_kTau - frank180_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c180_kTau, frank180_model.copulaModel.kTau(), delta=0.02)

        # 270 deg
        frank270 = Copula("frank", 3)
        u270, v270 = frank270.sample(10000, *(8.0,))
        g270 = sns.jointplot(u270, v270, stat_func=kendalltau)
        g270.savefig("frank_sample_pdf_270.png")
        frank270.fittedParams = (8.0,)
        c270_kTau = frank270.kTau()
        self.assertAlmostEqual(c90_kTau, -0.602619667, 6)
        # compute rank corr coeff from resampled data
        frank270_model = PairCopula(u270, v270, family={"frank": 3})
        frank270_model.copulaTournament()
        print(frank270_model.copulaParams)
        # check that frank copula won since
        # we seeded with samples from a frank df
        self.assertTrue("frank" in frank270_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c270_kTau - frank270_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c270_kTau, frank270_model.copulaModel.kTau(), delta=0.02)

    def testClaytonRotate(self):
        # 0 deg
        clayton00 = Copula("clayton", 0)
        u00, v00 = clayton00.sample(10000, *(8.0,))
        g00 = sns.jointplot(u00, v00, stat_func=kendalltau)
        g00.savefig("clayton_sample_pdf_00.png")
        clayton00.fittedParams = (8.0,)
        c00_kTau = clayton00.kTau()
        # check kTau
        # self.assertAlmostEqual(c00_kTau, 0.602619667, 6)
        # compute rank corr coeff from resampled data
        clayton00_model = PairCopula(u00, v00, family={"clayton": 0})
        clayton00_model.copulaTournament()
        print(clayton00_model.copulaParams)
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton00_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c00_kTau - clayton00_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.01)
        self.assertAlmostEqual(c00_kTau, clayton00_model.copulaModel.kTau(), delta=0.01)

        # 90 deg
        clayton90 = Copula("clayton", 1)
        u90, v90 = clayton90.sample(10000, *(8.0,))
        g90 = sns.jointplot(u90, v90, stat_func=kendalltau)
        g90.savefig("clayton_sample_pdf_90.png")
        clayton90.fittedParams = (8.0,)
        c90_kTau = clayton90.kTau()
        # self.assertAlmostEqual(c90_kTau, -0.602619667, 6)
        # compute rank corr coeff from resampled data
        clayton90_model = PairCopula(u90, v90, family={"clayton": 1})
        clayton90_model.copulaTournament()
        print(clayton90_model.copulaParams)
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton90_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c90_kTau - clayton90_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c90_kTau, clayton90_model.copulaModel.kTau(), delta=0.02)

        # 180 deg
        clayton180 = Copula("clayton", 2)
        u180, v180 = clayton180.sample(10000, *(8.0,))
        g180 = sns.jointplot(u180, v180, stat_func=kendalltau)
        g180.savefig("clayton_sample_pdf_180.png")
        clayton180.fittedParams = (8.0,)
        c180_kTau = clayton180.kTau()
        # self.assertAlmostEqual(c00_kTau, 0.602619667, 6)
        # compute rank corr coeff from resampled data
        clayton180_model = PairCopula(u180, v180, family={"clayton": 2})
        clayton180_model.copulaTournament()
        print(clayton180_model.copulaParams)
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton180_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c180_kTau - clayton180_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c180_kTau, clayton180_model.copulaModel.kTau(), delta=0.02)

        # 270 deg
        clayton270 = Copula("clayton", 3)
        u270, v270 = clayton270.sample(10000, *(8.0,))
        g270 = sns.jointplot(u270, v270, stat_func=kendalltau)
        g270.savefig("clayton_sample_pdf_270.png")
        clayton270.fittedParams = (8.0,)
        c270_kTau = clayton270.kTau()
        # self.assertAlmostEqual(c90_kTau, -0.602619667, 6)
        # compute rank corr coeff from resampled data
        clayton270_model = PairCopula(u270, v270, family={"clayton": 3})
        clayton270_model.copulaTournament()
        print(clayton270_model.copulaParams)
        # check that clayton copula won since
        # we seeded with samples from a clayton df
        self.assertTrue("clayton" in clayton270_model.copulaModel.name)
        # Ensure kTau is nearly the same from resampled data
        kTauDelta = c270_kTau - clayton270_model.copulaModel.kTau()
        self.assertTrue(abs(kTauDelta) < 0.04)
        # self.assertAlmostEqual(c270_kTau, clayton270_model.copulaModel.kTau(), delta=0.02)
