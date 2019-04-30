##
# \brief Clayton copula.
import numpy as np
from .copula_base import CopulaBase


class ClaytonCopula(CopulaBase):
    """!
    @brief Clayton copula.
    Single parameter model.
    \f[
        \theta \in [0, \infty)
    \f]
    """
    def __init__(self, rotation=0, init_params=None):
        self.thetaBounds = ((1e-9, np.inf),)
        self.theta0 = (1.0, )
        self.rotation = rotation
        self.name = 'clayton'
        super(ClaytonCopula, self).__init__(rotation, params=init_params)

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for frank bivariate copula
        """
        if theta[0] == 0:
            p = np.ones(np.asarray(u).size)
            return p
        else:
            h1 = (1 + 2.0 * theta[0]) / theta[0]
            h2 = 1.0 + theta[0]
            h3 = -theta[0]

            UU = np.asarray(u)
            VV = np.asarray(v)

            h4 = np.power(UU,h3)+np.power(VV,h3) - 1.0

            p = h2*np.power(UU,-h2)*np.power(VV,-h2)*np.power(h4,-h1)
            return p

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        h1 = -theta[0]
        h2 = 1. / h1

        UU = np.asarray(u)
        VV = np.asarray(v)

        hu = np.power(UU, h1)
        hv = np.power(VV, h1)

        p = np.power(hu + hv - 1.0, h2)
        return p

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        """
        TODO: CHECK UU and VV ordering!
        """
        h1 = -(1.0 + theta[0]) / theta[0]
        # UU = np.asarray(1. - u)
        # VV = np.asarray(1. - v)
        UU = np.asarray(u)
        VV = np.asarray(v)
        uu = np.power(np.power(VV,theta[0])*(np.power(UU,-theta[0])-1.0)+1.0,h1);
        return uu

    @CopulaBase._rotHinv
    def _hinv(self, v, u, rotation=0, *theta):
        """
        TODO: CHECK UU and VV ordering!
        """
        h1 = -1.0 / theta[0]
        h2 = -theta[0] / (1.0 + theta[0])
        UU = np.asarray(u)
        VV = np.asarray(v)
        uu = np.power(np.power(VV,-theta[0])*(np.power(UU,h2)-1.0)+1.0,h1);
        return uu

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        return (1.0 / theta[0]) * (np.power(t, -theta[0]) - 1.0)

    def _kTau(self, rotation=0, *theta):
        # return self._kTau(rotation, *theta)
        if self.rotation == 1 or self.rotation == 3:
            return - theta[0] / (theta[0] + 2)
        else:
            return theta[0] / (theta[0] + 2)
