##
# \brief Clayton copula.
import numpy as np
from copula_base import CopulaBase


class ClaytonCopula(CopulaBase):
    """!
    @brief Clayton copula.
    Single parameter model
    """
    def __init__(self):
        pass

    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for frank bivariate copula
        """
        if theta[0] == 0:
            p = np.ones(len(u))
            return p
        else:
            h1 = (1 + 2.0 * theta[0]) / theta[0]
            h2 = 1.0 + theta[0]
            h3 = -theta[0]

            UU = np.array(u)
            VV = np.array(v)

            h4 = np.power(UU,h3)+np.power(VV,h3) - 1.0

            p = h2*np.power(UU,-h2)*np.power(VV,-h2)*np.power(h4,-h1)
            return p

    def _h(self, u, v, rotation=0, *theta):
        h1 = -(1.0 + theta[0]) / theta[0]
        UU = np.array(u);
        VV = np.array(v);
        uu = np.power(np.power(VV,theta[0])*(np.power(UU,-theta[0])-1.0)+1.0,h1);
        return uu

    def _hinv(self, U, V, rotation=0, *theta):
        h1 = -1.0 / theta[0]
        h2 = -theta[0]/(1.0+theta[0])
        UU = np.array(U);
        VV = np.array(V);
        uu = np.power(np.power(VV,-theta[0])*(np.power(UU,h2)-1.0)+1.0,h1);
        return uu
