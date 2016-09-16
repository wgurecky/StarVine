##
# \brief Indipencene copula.
import numpy as np
from copula_base import CopulaBase


class IndepCopula(CopulaBase):
    def __init__(self):
        pass

    def _pdf(self, u, v, rotation_theta=0):
        return np.ones(len(u))

    def _h(self, u, v, rotation_theta=0):
        return u

    def _hinv(self, u, v, rotation_theta=0):
        return u

    def _gen(self, t, *theta):
        return -np.log(t)