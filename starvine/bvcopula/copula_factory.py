##
# \brief Copula factory
from __future__ import print_function, absolute_import, division
from starvine.bvcopula.copula import *
import re


def validateRotation(rotation):
    if rotation not in [0, 1, 2, 3]:
        raise RuntimeError("Invalid rotation.")


def Copula(copulatype, rotation=0):
    """!
    @brief Factory method. Returns a bivariate copula instance.
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
    elif re.match("oklin", copulatype):
        return gumbel_copula.OlkinCopula(rotation)
    elif re.match("indep", copulatype):
        return indep_copula.IndepCopula(rotation)
    else:
        # default
        raise RuntimeError("Copula type %s is not available" % str(copulatype))
