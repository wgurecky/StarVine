#!/usr/bin/python3
from __future__ import print_function, division
# starvine imports
import context
from starvine.vine.R_vine import Rvine
from starvine.mvar.mv_plot import matrixPairPlot
# extra imports
from scipy.stats import norm, beta
import unittest
import os
import numpy as np
import pandas as pd
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


class TestRvine(unittest.TestCase):
    def testRvineConstruct(self):
        stocks = np.loadtxt(dataDir + 'stocks.csv', delimiter=',')
        x = stocks[:, 0]
        y = stocks[:, 1]
        z = stocks[:, 4]
        p = stocks[:, 5]
        e = stocks[:, 6]
        # Create pandas data table
        tstData = pd.DataFrame()
        tstData['1a'] = x
        tstData['2b'] = y
        tstData['3c'] = z
        tstData['4d'] = p
        tstData['5e'] = e
        # Visualize multivar data
        matrixPairPlot(tstData, savefig="quad_varaite_ex.png")
        # Visualize multivar ranked data
        ranked_data = tstData.dropna().rank()/(len(tstData)+1)
        # ranked_data['1a'] = ranked_data['1a']
        matrixPairPlot(ranked_data, savefig="quad_varaite_ranked_ex.png")

        # Init Rvine
        tstVine = Rvine(ranked_data, trial_copula={"gauss": 0})

        # construct the vine
        tstVine.constructVine()

        # plot vine
        tstVine.plotVine(savefig="r_vine_graph_ex.png")

        # sample from vine
        r_vine_samples = tstVine.sample(n=8000)
        matrixPairPlot(r_vine_samples, savefig="r_vine_resampled_ex.png")

        # check that the original data has same correlation coefficients as re-sampled
        # data from the fitted c-vine
        tst_rho_matrix = ranked_data.corr(method='pearson')
        tst_ktau_matrix = ranked_data.corr(method='kendall')
        sample_rho_matrix = r_vine_samples.corr(method='pearson')
        sample_ktau_matrix = r_vine_samples.corr(method='kendall')
        # sort by col labels
        tst_rho_matrix = tst_rho_matrix.reindex(sorted(tst_rho_matrix.columns), axis=1)
        tst_ktau_matrix = tst_ktau_matrix.reindex(sorted(tst_ktau_matrix.columns), axis=1)
        sample_rho_matrix = sample_rho_matrix.reindex(sorted(sample_rho_matrix.columns), axis=1)
        sample_ktau_matrix = sample_ktau_matrix.reindex(sorted(sample_ktau_matrix.columns), axis=1)

        print("Original data corr matrix:")
        print(tst_rho_matrix)
        print("Vine sample corr matrix:")
        print(sample_rho_matrix)
        print("Diff:")
        print(tst_rho_matrix - sample_rho_matrix)
        self.assertTrue(np.allclose(tst_rho_matrix - sample_rho_matrix, 0, atol=0.10))
        self.assertTrue(np.allclose(tst_ktau_matrix - sample_ktau_matrix, 0, atol=0.10))

        # fit marginal distributions to original data
        marginal_dict = {}
        for col_name in tstData.columns:
            marginal_dict[col_name] = beta(*beta.fit(tstData[col_name]))
        # scale the samples
        r_vine_scaled_samples_a = tstVine.scaleSamples(r_vine_samples, marginal_dict)
        matrixPairPlot(r_vine_scaled_samples_a, savefig="r_vine_varaite_resampled_scaled_a.png")

        r_vine_scaled_samples_b = tstVine.sampleScale(8000, marginal_dict)

        # compute correlation coeffs
        sample_scaled_rho_matrix_a = r_vine_scaled_samples_a.corr(method='pearson')
        sample_scaled_rho_matrix_b = r_vine_scaled_samples_b.corr(method='pearson')

        # check for consistency
        self.assertTrue(np.allclose(tst_rho_matrix - sample_scaled_rho_matrix_a, 0, atol=0.1))
        self.assertTrue(np.allclose(tst_rho_matrix - sample_scaled_rho_matrix_b, 0, atol=0.1))


if __name__ == "__main__":
    unittest.main()
