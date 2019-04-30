##
# \brief Indipencene copula.
import numpy as np
from .copula_base import CopulaBase


class IndepCopula(CopulaBase):
    def __init__(self, rotation=0):
        self.name = 'indep'
        self.theta0 = (1.0, )
        self.fittedParams = [1.0]
        self.thetaBounds = ((-np.inf, np.inf),)

    def _pdf(self, u, v, rotation=0, *args):
        return np.ones(len(u))

    def _cdf(self, u, v, rotation=0, *args):
        return u * v

    def _h(self, u, v, rotation=0, *args):
        return v

    def _hinv(self, u, v, rotation=0, *args):
        return v

    def _gen(self, t, rotation=0, *theta):
        return -np.log(t)
