#!/usr/bin/env python2
from __future__ import print_function, division
import unittest
from scipy.optimize import bisect
from scipy.stats.mstats import rankdata
from scipy.stats import kendalltau
from scipy.stats import gaussian_kde
from copula.t_copula import StudentTCopula as stc
from copula.gauss_copula import GaussCopula as stg
from bv_plot import bvPairPlot
import pylab as pl
import numpy as np
import os
import seaborn as sns;
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)
tol = 0.1


class TestTcopulaFit(unittest.TestCase):
    def testTcoplulaFit(self):
        print("--------------------- T COPULA FIT TEST --------------------------")
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]

        # plot dataset for visual inspection
        marg_dict = {}
        plt0 = sns.jointplot(x, y, marginal_kws=marg_dict, stat_func=kendalltau)
        plt0.savefig("original_stocks.png")
        bvPairPlot(x, y, savefig="original_stocks_pair.png")

        # Rank transform the data
        u = rankdata(x) / (len(x) + 1)
        v = rankdata(y) / (len(y) + 1)
        plt1 = sns.jointplot(u, v, marginal_kws=marg_dict, stat_func=kendalltau)
        plt1.savefig("rank_transformed.png")

        # CDF tranformed data
        kde_x = gaussian_kde(x)
        kde_y = gaussian_kde(y)
        x_hat = np.zeros(len(x))
        y_hat = np.zeros(len(y))
        for i, (xp, yp) in enumerate(zip(x, y)):
            x_hat[i] = kde_x.integrate_box_1d(-np.inf, xp)
            y_hat[i] = kde_y.integrate_box_1d(-np.inf, yp)
        plt11 = sns.jointplot(x_hat, y_hat, marginal_kws=marg_dict, stat_func=kendalltau)
        plt11.savefig("cdf_transformed.png")

        # Fit t copula and gaussian copula
        thetat0 = [0.7, 30]
        thetag0 = [0.2]
        g_copula = stg()
        theta_g_fit = g_copula.fitMLE(u, v, 0, *thetag0, bounds=((-0.99, 0.99),))
        print("Gaussian copula MLE paramter [rho]: ", theta_g_fit)
        t_copula = stc()
        theta_t_fit = t_copula.fitMLE(u, v, 0, *thetat0, bounds=((-0.99, 0.99),(1, 1e8),))
        print("T copula MLE parameters [rho, DoF]: ", theta_t_fit)

        # For this dataset, the t-copula should approach the gaussian copula fit
        self.assertAlmostEqual(theta_g_fit[0], theta_t_fit[0], places=3)

        # Sample from the fitted gaussian copula and plot
        ug_hat, vg_hat = g_copula.sample(1000, 0, *theta_g_fit)
        pl.figure(2)
        plt2 = sns.jointplot(ug_hat, vg_hat, stat_func=kendalltau)
        plt2.savefig("gaussian_copula_hat.png")

        # Sample from the fitted t copula and plot
        ut_hat, vt_hat = t_copula.sample(1000, 0, *theta_t_fit)
        pl.figure(3)
        plt3 = sns.jointplot(ut_hat, vt_hat, stat_func=kendalltau)
        plt3.savefig("t_copula_hat.png")

        # Random noise
        rv = np.random.uniform(0, 1, 2000)
        ru = np.random.uniform(0, 1, 2000)
        plt4 = sns.jointplot(ru, rv, stat_func=kendalltau)
        plt4.savefig("rand_scatter.png")

        # Resample
        def icdf_uv_bisect(ux, X):
            icdf = np.zeros(np.array(X).size)
            for i, xx in enumerate(X):
                kde_f = gaussian_kde(ux)
                kde_cdf_err = lambda m: xx - kde_f.integrate_box_1d(-np.inf, m)
                try:
                    icdf[i] = bisect(kde_cdf_err,
                                     min(ux) - np.abs(0.5 * min(ux)),
                                     max(ux) + np.abs(0.5 * max(ux)),
                                     xtol=1e-6, maxiter=200)
                except:
                    icdf[i] = np.nan
            return icdf
        resampled_x = icdf_uv_bisect(x, ut_hat)
        resampled_y = icdf_uv_bisect(y, vt_hat)
        plt5 = sns.jointplot(resampled_x, resampled_y, stat_func=kendalltau)
        plt5.savefig("resampled_scatter.png")

        # Compare to expected results
        true_rho = 0.7220  # shape (related to pearsons corr coeff)
