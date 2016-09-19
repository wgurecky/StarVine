##
# \brief Gumbel copula.
import numpy as np
from copula_base import CopulaBase


class GumbelCopula(CopulaBase):
    """!
    @brief Gumbel copula
    single paramter model
    \f$\theta \in [1, \infty) \f$
    """
    def __init__(self):
        self.thetaBounds = ((1+1e-9, np.inf),)
        self.theta0 = (2.0, )
        self.name = 'gumbel'

    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for gumbel bivariate copula
        """
        h1 = theta[0] - 1.0
        # h2 = (1.0 - 2.0 ** theta[0]) / theta[0]
        h2 = (1.0 - 2.0 * theta[0]) / theta[0]
        h3 = 1.0 / theta[0]

        UU = np.array(u)
        VV = np.array(v)

        h4 = -np.log(UU)
        h5 = -np.log(VV)
        h6 = np.power(h4, theta[0]) + np.power(h5, theta[0])
        h7 = np.power(h6, h3)

        p = np.exp(-h7+h4+h5)*np.power(h4,h1)*np.power(h5,h1)*np.power(h6,h2)*(h1+h7)
        return p

    def _h(self, u, v, rotation=0, *theta):
        h1 = theta[0] - 1.0
        h2 = (1.0 - theta[0]) / theta[0]
        h3 = 1.0 / theta[0]

        UU = np.array(u)
        VV = np.array(v)

        h4 = -np.log(VV)
        h5 = np.power(-np.log(UU), theta[0]) + np.power(h4, theta[0])

        uu = np.power(h4,h1)/VV*(np.power(h5,h2))*np.exp(-np.power(h5,h3))
        return uu

    def _hinv(self, u, v, rotation=0, *theta):
        """!
        TODO: Computing hinv by bisection is slow. speed up needed!
        """
        U = np.array(u)
        V = np.array(v)
        uu = np.zeros(U.size)
        for i, (uu, vv) in enumerate(zip(U, V)):
            uu[i] = self._invhfun_bisect(uu, vv, rotation, *theta)
        return uu

    def _gen(self, t, *theta):
        return np.power(-np.log(t), theta[0])

    def kTau(self, rotation=0, *theta):
        return self._kTau(rotation, *theta)
