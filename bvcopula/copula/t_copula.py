##
# \brief Student's T copula.
import numpy as np
import scipy as sp
from scipy.special import gammaln
from copula_base import CopulaBase


class StudentTCopula(CopulaBase):
    """!
    @brief Student T copula
    2 parameter model

    theta[0] == rho (shape param, related to pearson's corr coeff)
    theta[1] == nu (degrees of freedom)
    \f$\theta[0] \in (-1, 1) \f$
    \f$\theta[1] \in (2, \infty) \f$
    """
    def __init__(self):
        self.thetaBounds = ((-1 + 1e-9, 1 - 1e-9), (2.0, np.inf),)
        self.theta0 = (0.7, 10.0)
        self.name = 't'

    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function of T copula.
        @param u <np_1darary>
        @param v <np_1darary>
        @param rotation <int>  Optional copula rotation.
        @param theta <list of float> list of parameters to T-copula
               [Shape, DoF]
        """
        # Constants
        rho2 = np.power(theta[0], 2.0)
        h1 = 1.0 - rho2
        h2 = theta[1] / 2.0
        h3 = h2 + 0.5
        h4 = h2 + 1.0
        h5 = 1.0 / theta[1]
        h6 = h5 / h1
        # T random var with theta[1] DoF parameter (unit SD, centered at 0)
        t_rv = sp.stats.t(df=theta[1], scale=1.0, loc=0.0)

        # u and v must be inside the unit square ie. in (0, 1)
        # clipMask = ((v < 1.0) & (v > 0.0) & (u < 1.0) & (v > 0.0))
        UU = np.array(u)
        VV = np.array(v)

        # Percentile point function eval
        x = t_rv.ppf(UU)
        y = t_rv.ppf(VV)

        x2 = np.power(x, 2.0)
        y2 = np.power(y, 2.0)

        p = ggamma(h4)*ggamma(h2)/np.sqrt(h1)/np.power(ggamma(h3),2)*np.power(1+h5*x2,h3)* \
            np.power(1+h5*y2,h3)/np.power(1+h6*(x2+y2-2*theta[0]*x*y),h4)
        if np.any(np.isinf(p)):
            print("WARNING: INF probability returned by PDF")
        return p


    def _h(self, u, v, rotation=0, *theta):
        """!
        @brief H function (Conditional distribution) of T copula.
        """
        h1 = 1.0 - np.power(theta[0], 2.0)
        nu1 = theta[1] + 1.0
        dist1 = sp.stats.t(df=theta[1], scale=1.0, loc=0.0)
        dist2 = sp.stats.t(df=nu1, scale=1.0, loc=0.0)

        UU = np.array(u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist1.ppf(UU)
        y = dist1.ppf(VV)

        # eval H function
        uu = dist2.cdf((x - theta[0] * y) / np.sqrt((theta[1] + np.power(y, 2)) * h1 / nu1))
        # todo check bounds of output should be in [0, 1]
        return uu


    def _hinv(self, u, v, rotation=0, *theta):
        """!
        @brief Inverse H function (Inv Conditional distribution) of T copula.
        """
        h1 = 1.0 - np.power(theta[0], 2.0)
        nu1 = theta[1] + 1.0
        dist1 = sp.stats.t(df=theta[1], scale=1.0, loc=0.0)
        dist2 = sp.stats.t(df=nu1, scale=1.0, loc=0.0)

        UU = np.array(u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist2.ppf(UU)
        y = dist1.ppf(VV)

        # eval H function
        uu = dist1.cdf(x * np.sqrt((theta[1] + np.power(y, 2.0)) * h1 / nu1) + theta[0] * y)
        return uu

    def kTau(self, *theta):
        return (2.0 / np.pi) * np.arcsin(theta[0])


def ggamma(x):
    return np.log(gammaln(x))
