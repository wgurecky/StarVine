#!/usr/bin/env python
#-*- coding: utf-8 -*-

from six import iteritems
import unittest
import numpy as np
from starvine.bvcopula import copula_factory
from scipy.stats import beta
# tpl
import dcor
# internal deps
from starvine.bvcopula.stat_tests import mardias_test, ks2d2s, estat2d, gauss_copula_test


class TestMiscStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alpha = 0.10
        cls.margin_1 = beta(2.0, 1.5)  # skewed beta rv
        cls.margin_2 = beta(1.5, 2.2)  # skewed beta rv

    def testMardias(self):
        # rand seed
        np.random.seed(123)
        test_data = \
            np.array([[2.4,2.1,2.4],[3.5,1.8,3.9],[6.7,3.6,5.9],[5.3,3.3,6.1],[5.2,4.1,6.4], \
            [3.2,2.7,4.0],[4.5,4.9,5.7],[3.9,4.7,4.7],[4.0,3.6,2.9],[5.7,5.5,6.2],[2.4,2.9,3.2], \
            [2.7,2.6,4.1]])

        # hypoth test results
        p1, p1c, p2, h_dict = mardias_test(test_data, cov_bias=True)

        # check results
        self.assertAlmostEqual(p1, 0.9171, delta=0.001)
        self.assertAlmostEqual(p1c, 0.7735, delta=0.001)
        self.assertAlmostEqual(p2, 0.2653, delta=0.001)

        # generate trivariate gaussian data
        mvar_norm_data = \
            np.random.multivariate_normal([0.0, 0.0, 0.0], cov=h_dict['cov'], size=1000)

        # ensure we accept h0 (this data is def mvar gaussian)
        p1, p1c, p2, h_dict = mardias_test(mvar_norm_data, cov_bias=True, alpha=0.05)
        self.assertTrue(h_dict['h0'])
        self.assertTrue(h_dict['skew_h0'])
        self.assertTrue(h_dict['kurt_h0'])

        # generate bivar data with normal copula but non-normal margins
        margin_1 = beta(2.0, 1.5)  # skewed beta rv
        margin_2 = beta(1.5, 2.2)  # skewed beta rv

        # setup copula
        copula = copula_factory.Copula('gauss')
        copula.fitKtau(0.8)  # strong positive dep structure

        # draw samples
        x1, x2 = copula.sampleScale(margin_1, margin_2, n=1000)
        non_norm_data = np.array([x1, x2]).T

        # this should fail the Mardias test for multi-normality
        p1, p1c, p2, h_dict = mardias_test(non_norm_data, cov_bias=True)
        self.assertFalse(h_dict['h0'])
        self.assertFalse(h_dict['skew_h0'])
        self.assertFalse(h_dict['kurt_h0'])


    def testBivarKS(self):
        """
        @brief Tests bivariate Kolmogorov-Smirnov test.
        """
        # change rand seed
        print('---')
        np.random.seed(123)

        # setup copula dist 1
        copula_1 = copula_factory.Copula('gauss')
        copula_1.fitKtau(0.5)
        x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=400)
        # x1, y1 = copula_1.sample(n=400)

        # setup copula dist 2
        copula_2 = copula_factory.Copula('gauss')
        copula_2.fitKtau(0.5)
        x2, y2 = copula_2.sampleScale(self.margin_1, self.margin_2, n=400)
        # x2, y2 = copula_2.sample(n=400)

        # perform bivar 2 sample KS test
        ks_p, D = ks2d2s(x1, y1, x2, y2, nboot=1000)
        en_h = dcor.homogeneity.energy_test(np.array([x1, y1]).T,
                                            np.array([x2, y2]).T, num_resamples=1000)
        en_p, en_s = en_h.p_value, en_h.statistic
        print((ks_p, en_p))
        # self.assertGreater(ks_p, self.alpha)

        # check in case of negative dep structure
        ks_p, D = ks2d2s(-x1, y1, -x2, y2, nboot=1000)
        en_h = dcor.homogeneity.energy_test(np.array([-x1, y1]).T,
                                            np.array([-x2, y2]).T, num_resamples=1000)
        en_p, en_s = en_h.p_value, en_h.statistic
        print((ks_p, en_p))
        self.assertGreater(ks_p, self.alpha)

        # Check in case of two different copula.
        # Assumes distributions with equal marginal distributions
        # but a different copula
        copula_1 = copula_factory.Copula('clayton', rotation=1)
        copula_1.fitKtau(-0.4)
        x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=400)
        # x1, y1 = copula_1.sample(n=400)
        copula_2 = copula_factory.Copula('gauss')
        copula_2.fitKtau(-0.7)
        x2, y2 = copula_2.sampleScale(self.margin_1, self.margin_2, n=400)
        # x2, y2 = copula_2.sample(n=400)
        ks_p, D = ks2d2s(x1, y1, x2, y2, nboot=1000)
        en_h = dcor.homogeneity.energy_test(np.array([x1, y1]).T,
                                            np.array([x2, y2]).T, num_resamples=1000)
        en_p, en_s = en_h.p_value, en_h.statistic
        print((ks_p, en_p))
        # should accept null hypoth that data are from different dists
        self.assertLess(ks_p, self.alpha)

    def testGaussCopulaTest(self):
        np.random.seed(123)
        # check the gauss hypothesis is false when given clayton orig data
        print('--- clayton')
        for i in range(3):
            copula_1 = copula_factory.Copula('clayton', rotation=1)
            copula_1.fitKtau(-0.55)
            x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=2000)
            g_p, _, g_h = gauss_copula_test(x1, y1, dist='ad-avg', procs=6)
            print((g_p, g_h['h0']))
            self.assertFalse(g_h['h0'])
        print('--- frank')
        for i in range(3):
            copula_1 = copula_factory.Copula('frank', rotation=1)
            copula_1.fitKtau(-0.55)
            x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=2000)
            g_p, _, g_h = gauss_copula_test(x1, y1, dist='ad-avg', procs=6)
            print((g_p, g_h['h0']))
            self.assertFalse(g_h['h0'])
        print('--- gumbel')
        for i in range(3):
            copula_1 = copula_factory.Copula('gumbel', rotation=1)
            copula_1.fitKtau(-0.55)
            # WARNING: Gumbel seems to be tough to distinguish from gauss
            # when |ktau| < 0.5.  A larger population sample size helps
            x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=8000)
            g_p, _, g_h = gauss_copula_test(x1, y1, dist='ad-avg', procs=6)
            print((g_p, g_h['h0']))
            self.assertFalse(g_h['h0'])
        print('--- gauss')
        # check the gauss hypothesis is true when given gauss orig data
        for i in range(3):
            copula_1 = copula_factory.Copula('gauss', rotation=0)
            copula_1.fitKtau(-0.55)
            x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=2000)
            g_p, _, g_h = gauss_copula_test(x1, y1, dist='ad-avg', procs=6)
            print((g_p, g_h['h0']))
            self.assertTrue(g_h['h0'])

    def testGaussCopulaTestSweep(self):
        print('=== Ktau sweep')
        # plot p-val for a variety of kendall's tau vals
        ktaus = np.linspace(1e-2, 0.9, 10)
        for ktau in ktaus:
            copula_1 = copula_factory.Copula('gauss', rotation=1)
            copula_1.fitKtau(ktau)
            x1, y1 = copula_1.sampleScale(self.margin_1, self.margin_2, n=500)
            g_p, _, g_h = gauss_copula_test(x1, y1, dist='ks-avg', procs=6)
            print((ktau, g_p, g_h['h0']))
            self.assertTrue(g_h['h0'])
