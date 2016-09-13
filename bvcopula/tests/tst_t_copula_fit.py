#!/usr/bin/env python2

from __future__ import print_function, division
from uvmodels.uv_beta import UVBeta
from scipy.stats.mstats import rankdata
import pylab as pl
import numpy as np
import unittest
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
        # Compare to expected results
        true_rho = 0.7220
        true_nu = 3.18e6
