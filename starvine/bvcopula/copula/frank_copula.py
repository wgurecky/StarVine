##
# \brief Frank copula.
import numpy as np
from numba import jit
from scipy.integrate import quad
from copula_base import CopulaBase


class FrankCopula(CopulaBase):
    """!
    @brief Frank copula.
    Single parameter
    \f$\theta \in [0, \infty) \f$
    """
    def __init__(self, rotation=0, init_params=None):
        self.thetaBounds = ((1e-9, np.inf),)
        self.theta0 = (1.0,)
        self.rotation = rotation
        self.name = 'frank'
        super(FrankCopula, self).__init__(rotation, params=init_params)

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for frank bivariate copula
        """
        if theta[0] == 0:
            p = np.ones(np.asarray(u).size)
            return p
        else:
            h1 = -theta[0]
            h2 = expm1(h1)
            h3 = h1 * h2

            UU = np.asarray(u)
            VV = np.asarray(v)

            h4 = expm1(h1 * UU) * expm1(h1 * VV)
            p = h3 * np.exp(h1 * (UU + VV)) / np.power(h2 + h4, 2.0)
            return p

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        h1 = -theta[0]
        h2 = expm1(h1)

        UU = np.asarray(u)
        VV = np.asarray(v)
        h3 = expm1(h1 * UU) * expm1(h1 * VV)

        p = -np.log(1.0 + h3 / h2) / theta[0]
        return p

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        """
        TODO: CHECK UU and VV ordering!
        """
        h1 = np.exp(-theta[0])
        UU = np.asarray(u)
        VV = np.asarray(v)
        h2 = np.power(h1, UU)
        h3 = np.power(h1, VV)
        h4 = h2 * h3
        uu = (h4 - h3) / (h4 - h2 - h3 + h1)
        return uu

    @CopulaBase._rotHinv
    def _hinv(self, V, U, rotation=0, *theta):
        """
        TODO: CHECK UU and VV ordering!
        """
        h1 = np.exp(-theta[0])
        h2 = expm1(-theta[0])
        h3 = -1.0 / theta[0]
        UU = np.asarray(U, dtype=np.longdouble)
        VV = np.asarray(V)
        h4 = np.power(h1, VV, dtype=np.longdouble)

        # denom = (h4 * (1. / UU - 1.) + 1.)
        # log_arg = h2 / denom
        # log_arg2 = np.divide(h2, denom, dtype=np.longdouble)
        # uu2 = h3 * np.log(1. + log_arg2, dtype=np.longdouble)
        uu = h3 * np.log(1 + h2 / (h4 * (1 / UU - 1) + 1))
        if not (np.max(uu) <= 1.0) or not (np.min(uu) >= 0.0):
            uu = np.clip(uu, 1e-12, 1. - 1e-12)
        return np.asarray(uu, dtype=np.float64)

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        return -np.log((np.exp(-theta[0] * t) - 1.0) / (np.exp(-theta[0]) - 1.0))

    def _kTau(self, rotation=0, *theta):
        """!
        @brief Kendall's tau for frank copula.
            ref: Estimators for Archimedean copula in high dimensions.
            M. Hofert. et al.
            url: https://arxiv.org/pdf/1207.1708.pdf
        @param rotation copula rotaion parameter
        @param theta copula shape parameter list. Should have len==1
        """
        tau = 1. + (4. / theta[0]) * (debye_1(theta[0]) - 1.0)
        if self.rotation == 1 or self.rotation == 3:
            return -tau
        else:
            return tau


def debye_1(theta):
    """
    @brief Debye function of the first kind.
    """
    debye_int, err = quad(debye_exp_fn, 0.0, theta)
    return (1.0 / theta) * debye_int


@jit(nopython=True)
def debye_exp_fn(t):
    return t / (np.exp(t) - 1.0)


# @jit(nopython=True)
def expm1(x):
    """!
    @brief exponential - 1.0 helper
    """
    return np.exp(x, dtype=np.longdouble) - 1.0
