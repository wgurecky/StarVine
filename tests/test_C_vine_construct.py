#!/usr/bin/python2

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
        samples = tstVine.sample(n=8000)
        matrixPairPlot(samples, savefig="quad_varaite_resampled_ex.png")

