#!/usr/bin/python2
from __future__ import print_function, division
# starvine imports
import context
from starvine.vine.C_vine import Cvine
from starvine.mvar.mv_plot import matrixPairPlot
# extra imports
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
        e = stocks[:, 2]
        z = stocks[:, 4]
        #p = stocks[:, 5]
        # Create pandas data table
        tstData = pd.DataFrame()
        tstData['3a'] = x
        tstData['2b'] = y
        tstData['1c'] = z
        #tstData[3] = e
        #tstData[4] = p
        # Visualize multivar data
        matrixPairPlot(tstData, savefig="quad_varaite_ex.png")
        # Visualize multivar ranked data
        ranked_data = tstData.dropna().rank()/(len(tstData)+1)
        matrixPairPlot(ranked_data, savefig="quad_varaite_ranked_ex.png")

        # Init Cvine
        tstVine = Cvine(tstData)

        # construct the vine
        tstVine.constructVine()

        # plot vine
        tstVine.plotVine(savefig="c_vine_graph_ex.png")

        # sample from vine
        samples = tstVine.sample(n=5000)
        matrixPairPlot(samples, savefig="quad_varaite_resampled_ex.png")


if __name__ == "__main__":
    unittest.main()
