#!/usr/bin/env python2
from __future__ import print_function, division
import unittest
from scipy.stats.mstats import rankdata
from copula.t_copula import StudentTCopula as stc
import pylab as pl
import numpy as np
import os
pwd_ = os.getcwd()
dataDir = pwd_ + "/tests/data/"
np.random.seed(123)
tol = 0.1


class TestTcopulaFit(unittest.TestCase):
    def testTcoplulaFit(self):
        # Load matlab data set
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]

        # plot dataset for visual inspection
        pl.figure(0)
        pl.scatter(x, y)
        pl.savefig("original_stocks.png")

        # Rank transform the data
        u = rankdata(x) / len(x)
        v = rankdata(y) / len(y)
        pl.figure(1)
        pl.scatter(u, v)
        pl.savefig("rank_transformed.png")

        # Fit t copula
        # theta0 = [0.7, 3.0e6]
        theta0 = [0.7, 30]
        t_copula = stc()
        theta_fit = t_copula.fitMLE(u, v, 0, *theta0)

        # Compare to expected results
        true_rho = 0.7220  # shape (related to pearsons corr coeff)
        true_nu = 3.18e6   # DoF
