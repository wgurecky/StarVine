##
# \brief Gaussian copula (special case of t-copula where DoF = \inf)
import numpy as np
from scipy import stats
# STARVINE IMPORTS
from copula_base import CopulaBase
from mvtdstpack import mvtdstpack as mvt


class GaussCopula(CopulaBase):
    """!
    @brief Gaussian copula model
    single parameter
    \f$\theta[0] \in (-1, 1)\f$
    """
    def __init__(self, rotation=0, **kwargs):
        self.thetaBounds = ((-1 + 1e-9, 1 - 1e-9),)
        self.theta0 = (0.7,)
        self.name = 'gauss'
        self.rotation = rotation
        super(GaussCopula, self).__init__(rotation, **kwargs)

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function of Gauss copula.
        @param u <b>np_1darary</b>
        @param v <b>np_1darary</b>
        @param rotation <b>int</b>  Optional copula rotation.
        @param theta  Gaussian copula parameter
        """
        # Constants
        rho2 = np.power(theta[0], 2.0)
        h1 = 1.0 - rho2
        h2 = rho2 / (2.0 * h1)
        h3 = theta[0] / h1
        norm_rv = stats.norm(scale=1.0, loc=0.0)

        # UU = CheckBounds(u);
        # VV = CheckBounds(v);
        # u and v must be on the unit square ie. in [0, 1]
        UU = np.array(u)  # TODO: check bounds
        VV = np.array(v)

        # Output storage
        p = np.zeros(UU.size)

        # Percentile point function eval
        x = norm_rv.ppf(UU)
        y = norm_rv.ppf(VV)

        p = np.exp(h3 * x * y - h2 * (np.power(x, 2) + np.power(y, 2))) / np.sqrt(h1)
        return p

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        rho = theta[0]
        dof = 0
        norm_rv = stats.norm(scale=1.0, loc=0.0)

        UU = np.array(u)
        VV = np.array(v)

        # Output storage
        p = np.zeros(UU.size)

        lower = np.zeros((UU.size, 2))
        upper = np.zeros((UU.size, 2))
        upper[:, 0] = norm_rv.ppf(UU)
        upper[:, 1] = norm_rv.ppf(VV)
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
        @brief H function (Conditional distribution) of Gauss copula.
        TODO: CHECK UU and VV ordering!
        """
        kT = self.kTau(0, *theta)
        kTs = kT / abs(kT)
        kTM = 1 if kTs < 0 else 0

        h1 = np.sqrt(1.0 - np.power(np.array(theta[0]), 2))
        dist = stats.norm(scale=1.0, loc=0.0)

        UU = np.array(kTM + kTs * u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist.ppf(UU)
        y = dist.ppf(VV)

        # eval H function
        uu = dist.cdf((x - theta[0] * y) / h1)
        return uu

    @CopulaBase._rotHinv
    def _hinv(self, v, u, rotation=0, *theta):
        """!
        @brief Inverse H function (Inv Conditional distribution) of Gauss copula.
        TODO: CHECK UU and VV ordering!
        """
        kT = self.kTau(0, *theta)
        kTs = kT / abs(kT)
        kTM = 1 if kTs < 0 else 0

        h1 = np.sqrt(1.0 - np.power(np.array(theta[0]), 2))
        dist = stats.norm(scale=1.0, loc=0.0)

        UU = np.array(kTM + kTs * u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist.ppf(UU)
        y = dist.ppf(VV)

        # eval H function
        uu = dist.cdf(x * h1 + theta[0] * y)
        return uu

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        """!
        @brief Copula generating function
        """
        raise NotImplementedError

    def _kTau(self, rotation=0, *theta):
        return (2.0 / np.pi) * np.arcsin(theta[0])
