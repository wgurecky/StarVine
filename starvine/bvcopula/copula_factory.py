##
# \brief Concrete Copula factory class
from copula import *
import sys


class Copula(object):
    def __new__(cls, copulatype, rotation=0):
        if copulatype is "t":
            return t_copula.StudentTCopula(rotation)
        elif copulatype is "gauss":
            return gauss_copula.GaussCopula( rotation)
        elif copulatype is "frank":
            return frank_copula.FrankCopula( rotation)
        elif copulatype is "clayton":
            return clayton_copula.ClaytonCopula( rotation)
        elif copulatype is "gumbel":
            return gumbel_copula.GumbelCopula( rotation)
        elif copulatype is "indep":
            return indep_copula.IndepCopula( rotation)
        else:
            # default
            sys.exit("Invalid copula name.")

    def __init__(self, copulatype, rotation=0):
        self.rotation = rotation
        self.copulatype = copulatype
