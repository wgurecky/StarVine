##
# \brief Copula factory
from __future__ import print_function, absolute_import, division
from starvine.bvcopula.copula import *
import re


def validateRotation(rotation):
    if rotation not in [0, 1, 2, 3]:
        raise RuntimeError("Invalid rotation.")


def validateMixtureParams(mix_list):
    for tuple_copula_settings in mix_list:
        # ensure ("string", int, float) tuples in mix list
        assert len(tuple_copula_settings) == 3
        assert isinstance(tuple_copula_settings[0], str)  # type
        assert isinstance(tuple_copula_settings[1], int)  # rotation
        assert isinstance(tuple_copula_settings[2], float)  # wgt


def Copula(copulatype, rotation=0):
    """!
    @brief Factory method. Returns a bivariate copula instance.
    @param copulatype <b>string</b> Copula type.
    @param rotation <b>int</b>  Copula rotation 1==90 deg, 2==180 deg, 3==270 deg
    """
    if isinstance(copulatype, list):
        if len(copulatype) == 1:
            copulatype, rotation = copulatype[0], copulatype[1]
        elif len(copulatype) == 2:
            validateMixtureParams(copulatype)
            return MixCopula(*copulatype[0], *copulatype[1])
        else:
            raise RuntimeError("Mixture copula of more than 2 distributions is not implemented")
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


def MixCopula(copulatype_a, rotation_a=0, wgt_a=0.5, copulatype_b=None, rotation_b=0, wgt_b=0.5):
    """!
    @brief Helper method to generate a mixture distribution of two copula
    @param copulatype_a <b>string</b> Copula A type.
    @param rotation_a <b>int</b>  Copula rotation 1==90 deg, 2==180 deg, 3==270 deg
    @param copulatype_b <b>string</b> Copula B type.
    @param rotation_b <b>int</b>  Copula rotation 1==90 deg, 2==180 deg, 3==270 deg
    @param wgt_a <b>float</b>  Copula A weight in mixture
    @param wgt_b <b>float</b>  Copula B weight in mixture
    """
    wgt_a = wgt_a / (wgt_a + wgt_b)
    wgt_b = wgt_b / (wgt_a + wgt_b)
    return mixture_copula.MixtureCopula(Copula(copulatype_a, rotation_a), wgt_a,
                                        Copula(copulatype_b, rotation_b), wgt_b)
