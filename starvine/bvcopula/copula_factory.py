##
# \brief Concrete Copula factory class
from copula import *
import re
import sys


def validateRotation(rotation):
    if rotation not in [0, 1, 2, 3]:
        print("Invalid rotation.")
        raise RuntimeError


class Copula(object):
    """!
    @brief Returns a bivariate copula instance.
    """
    def __new__(cls, copulatype, rotation=0):
        """!
        @param copulatype <b>string</b> Copula type.
        @param rotation <b>int</b>  Copula rotation 1==90 deg, 2==180 deg, 3==270 deg
        """
        validateRotation(rotation)
        if re.match("t", copulatype):
            return t_copula.StudentTCopula(0)
        elif re.match("gauss", copulatype):
            return gauss_copula.GaussCopula(0)
        elif re.match("frank", copulatype):
            return frank_copula.FrankCopula(rotation)
        elif re.match("clayton", copulatype):
            return clayton_copula.ClaytonCopula(rotation)
        elif re.match("gumbel", copulatype):
            return gumbel_copula.GumbelCopula(rotation)
        elif re.match("indep", copulatype):
            return indep_copula.IndepCopula(rotation)
        else:
            # default
            sys.exit("Invalid copula name.")

    def __init__(self, copulatype, rotation=0):
        self.rotation = rotation
        self.copulatype = copulatype
