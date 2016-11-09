from __future__ import print_function, division
import unittest
from pc_base import PairCopula
from copula_factory import Copula


class TestSampleCopula(unittest.TestCase):
    def testGaussSample(self):
        pass

    def testTSample(self):
        pass

    def testClaytonSample(self):
        # 0 deg
        clayton00 = Copula("clayton", 0)
        u00, v00 = clayton00.sample(1000, *(0.5,))
        # compute rank corr coeff
        clayton00_model = PairCopula(u00, v00)
        # 90 deg
        clayton90 = Copula("clayton", 1)
        u90, v90 = clayton90.sample(1000, *(0.5,))
        # 180 deg
        clayton180 = Copula("clayton", 2)
        u180, v180 = clayton180.sample(1000, *(0.5,))
        # 270 deg
        clayton270 = Copula("clayton", 3)
        u270, v270 = clayton270.sample(1000, *(0.5,))

    def testGumbelSample(self):
        pass

    def testFrankSample(self):
        pass
