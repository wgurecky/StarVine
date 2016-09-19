##
# \brief Gaussian copula (special case of t-copula where DoF = \inf)
import numpy as np
import scipy as sp
from copula_base import CopulaBase


class GaussCopula(CopulaBase):
    """!
    @brief Gaussian copula model
    single parameter
    \f$\theta[0] \in (-1, 1)\f$
    """
    def __init__(self):
        self.thetaBounds = ((-1+1e-9, 1-1e-9),)
        self.theta0 = (0.7,)
        self.name = 'gauss'

    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function of Gauss copula.
        @param u <np_1darary>
        @param v <np_1darary>
        @param rotation <int>  Optional copula rotation.
        @param theta  Gaussian copula parameter
            [Shape, DoF]
        """
        # Constants
        rho2 = np.power(theta[0], 2.0)
        h1 = 1-rho2
        h2 = rho2 / (2.0 * h1)
        h3 = theta[0] / h1
        norm_rv = sp.stats.norm(scale=1.0, loc=0.0)

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

        p = np.exp(h3 * x  * y - h2 * (np.power(x, 2) + np.power(y, 2))) / np.sqrt(h1)
        return p


    def _h(self, u, v, rotation=0, *theta):
        """!
        @brief H function (Conditional distribution) of Gauss copula.
        """
        h1 = np.sqrt(1.0 - np.power(np.array(theta[0]), 2))
        dist = sp.stats.norm(scale=1.0, loc=0.0)

        UU = np.array(u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist.ppf(UU)
        y = dist.ppf(VV)

        # eval H function
        uu = dist.cdf((x - theta[0] * y) / h1)
        return uu


    def _hinv(self, u, v, rotation=0, *theta):
        """!
        @brief Inverse H function (Inv Conditional distribution) of Gauss copula.
        """
        h1 = np.sqrt(1.0 - np.power(np.array(theta[0]), 2))
        dist = sp.stats.norm(scale=1.0, loc=0.0)

        UU = np.array(u)  # TODO: check input bounds
        VV = np.array(v)

        # inverse CDF yields quantiles
        x = dist.ppf(UU)
        y = dist.ppf(VV)

        # eval H function
        uu = dist.cdf(x * h1 + theta[0] * y)
        return uu

    def _gen(self, t, *theta):
        """!
        @brief Copula generating function
        """
        raise NotImplementedError

    def kTau(self, *theta):
        return (2.0 / np.pi) * np.arcsin(theta[0])
