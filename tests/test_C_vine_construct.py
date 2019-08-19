#!/usr/bin/python2
from __future__ import print_function, division
# starvine imports
import context
from starvine.vine.C_vine import Cvine
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


class TestCvine(unittest.TestCase):
    def testCvineConstruct(self):
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

        # Init Cvine
        tstVine = Cvine(ranked_data)

        # construct the vine
        tstVine.constructVine()

        # plot vine
        tstVine.plotVine(savefig="c_vine_graph_ex.png")

        # sample from vine
        c_vine_samples = tstVine.sample(n=8000)
        matrixPairPlot(c_vine_samples, savefig="vine_resampled_ex.png")

        # check that the original data has same correlation coefficients as re-sampled
        # data from the fitted c-vine
        tst_rho_matrix = ranked_data.corr(method='pearson')
        tst_ktau_matrix = ranked_data.corr(method='kendall')
        sample_rho_matrix = c_vine_samples.corr(method='pearson')
        sample_ktau_matrix = c_vine_samples.corr(method='kendall')
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
        c_vine_scaled_samples_a = tstVine.scaleSamples(c_vine_samples, marginal_dict)
        matrixPairPlot(c_vine_scaled_samples_a, savefig="vine_varaite_resampled_scaled_a.png")

        c_vine_scaled_samples_b = tstVine.sampleScale(8000, marginal_dict)

        # compute correlation coeffs
        sample_scaled_rho_matrix_a = c_vine_scaled_samples_a.corr(method='pearson')
        sample_scaled_rho_matrix_b = c_vine_scaled_samples_b.corr(method='pearson')

        # check for consistency
        self.assertTrue(np.allclose(tst_rho_matrix - sample_scaled_rho_matrix_a, 0, atol=0.1))
        self.assertTrue(np.allclose(tst_rho_matrix - sample_scaled_rho_matrix_b, 0, atol=0.1))

        # check vine pdf values
        tstX = pd.DataFrame()
        tstX['1a'] = [0.01, 0.4, 0.5, 0.6, 0.99, 0.999]
        tstX['2b'] = [0.01, 0.4, 0.5, 0.6, 0.99, 0.999]
        tstX['3c'] = [0.01, 0.4, 0.5, 0.6, 0.99, 0.999]
        tstX['4d'] = [0.01, 0.4, 0.5, 0.6, 0.99, 0.999]
        tstX['5e'] = [0.01, 0.4, 0.5, 0.6, 0.99, 0.999]
        pdf_at_tstX = tstVine.vinePdf(tstX)
        print(pdf_at_tstX)
        self.assertTrue(np.all(pdf_at_tstX > 0.0))

        # check vine cdf values
        cdf_at_tstX = tstVine.vineCdf(tstX)
        print(cdf_at_tstX)
        cdf_tol = 0.05
        self.assertTrue(cdf_at_tstX[1] > cdf_at_tstX[0])
        self.assertTrue(np.all(0.0 <= cdf_at_tstX))
        self.assertTrue(np.all(cdf_at_tstX <= 1.0 + cdf_tol))
        self.assertAlmostEqual(cdf_at_tstX[-1], 1.0, delta=cdf_tol)


if __name__ == "__main__":
    unittest.main()
