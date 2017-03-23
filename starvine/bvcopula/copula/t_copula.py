##
# \brief Student's T copula.
import numpy as np
from scipy import stats
from scipy.special import gammaln
# STARVINE IMPORTS
from copula_base import CopulaBase
from mvtdstpack import mvtdstpack as mvt


class StudentTCopula(CopulaBase):
    """!
    @brief Student T copula
    2 parameter model

    \f$ \theta_0 == \rho \f$ (shape param, related to pearson's corr coeff)

    \f$ \theta_1 == \nu \f$  (degrees of freedom)

    \f$ \theta_0 \in (-1, 1), \f$
    \f$ \theta_1 \in (2, \infty) \f$
    """
    def __init__(self, rotation=0, init_params=None):
        self.thetaBounds = ((-1 + 1e-9, 1 - 1e-9), (2.0, np.inf),)
        self.theta0 = (0.7, 10.0)
        self.name = 't'
        self.rotation = 0
        super(StudentTCopula, self).__init__(rotation, params=init_params)

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function of T copula.
        @param u <b>np_1darary</b>
        @param v <b>np_1darary</b>
        @param rotation <b>int</b>  Optional copula rotation.
        @param theta <b>list of float</b> list of parameters to T-copula
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
        t_rv = stats.t(df=theta[1], scale=1.0, loc=0.0)

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

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        rho = theta[0]
        dof = int(round(theta[1]))
        t_rv = stats.t(df=theta[1], scale=1.0, loc=0.0)

        UU = np.array(u)
        VV = np.array(v)

        # Output storage
        p = np.zeros(UU.size)

        lower = np.zeros((UU.size, 2))
        upper = np.zeros((UU.size, 2))
        upper[:, 0] = t_rv.ppf(UU)
        upper[:, 1] = t_rv.ppf(VV)
        for i in range(UU.size):
            lowerb = lower[i, :]
            upperb = upper[i, :]
            inFin = np.zeros(upperb.size, dtype='int')     # integration limit setting
            delta = np.zeros(upperb.size, dtype='double')  # non centrality params
            error, value, status = mvt.mvtdst(dof, lowerb, upperb, inFin, rho, delta)
            p[i] = value
        return p

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        """!
        @brief H function (Conditional distribution) of T copula.
        TODO: CHECK UU and VV ordering!
        """
        kT = self.kTau(*theta)
        kTs = kT / abs(kT)
        kTM = 1 if kTs < 0 else 0

        h1 = 1.0 - np.power(theta[0], 2.0)
        nu1 = theta[1] + 1.0
        dist1 = stats.t(df=theta[1], scale=1.0, loc=0.0)
        dist2 = stats.t(df=nu1, scale=1.0, loc=0.0)

        UU = np.array(kTM + kTs * u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist1.ppf(UU)
        y = dist1.ppf(VV)

        # eval H function
        uu = dist2.cdf((x - theta[0] * y) / np.sqrt((theta[1] + np.power(y, 2)) * h1 / nu1))
        # todo check bounds of output should be in [0, 1]
        return uu

    @CopulaBase._rotHinv
    def _hinv(self, v, u, rotation=0, *theta):
        """!
        @brief Inverse H function (Inv Conditional distribution) of T copula.
        TODO: CHECK UU and VV ordering!
        """
        kT = self.kTau(rotation, *theta)
        kTs = kT / abs(kT)
        kTM = 1 if kTs < 0 else 0

        h1 = 1.0 - np.power(theta[0], 2.0)
        nu1 = theta[1] + 1.0
        dist1 = stats.t(df=theta[1], scale=1.0, loc=0.0)
        dist2 = stats.t(df=nu1, scale=1.0, loc=0.0)

        UU = np.array(kTM + kTs * u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist2.ppf(UU)
        y = dist1.ppf(VV)

        # eval H function
        uu = dist1.cdf(x * np.sqrt((theta[1] + np.power(y, 2.0)) * h1 / nu1) + theta[0] * y)
        return uu

    def _kTau(self, rotation=0, *theta):
        kt = (2.0 / np.pi) * np.arcsin(theta[0])
        return kt


def ggamma(x):
    return np.log(gammaln(x))
