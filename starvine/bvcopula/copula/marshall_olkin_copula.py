#!/usr/bin/python2
"""!
@brief Marshall-Olkin copula.
"""
from __future__ import print_function, absolute_import, division
import numpy as np
from starvine.bvcopula.copula.copula_base import CopulaBase


class OlkinCopula(CopulaBase):
    """!
    @brief Gumbel copula
    single paramter model
    \f$\theta \in [1, \infty) \f$
    """
    def __init__(self, rotation=0, init_params=None):
        self.thetaBounds = ((0, 1.), (0, 1.),)
        self.theta0 = (0.5, 0.7)
        self.rotation = rotation
        self.name = 'olkin'
        super(OlkinCopula, self).__init__(rotation, params=init_params)

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for gumbel bivariate copula
        """
        UU = np.asarray(u)
        VV = np.asarray(v)

        h3 = UU ** theta[0]
        h4 = VV ** theta[1]

        h3Mask = (h3 >= h4)
        h4Mask = (h3 <= h4)

        # evaluate PDF
        p = np.zeros(len(UU))
        p[h3Mask] = (1. - theta[0]) * UU[h3Mask] ** (-theta[0])
        p[h4Mask] = (1. - theta[1]) * VV[h4Mask] ** (-theta[1])
        return p

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        UU = np.asarray(u)
        VV = np.asarray(v)

        h3 = UU ** theta[0]
        h4 = VV ** theta[1]

        h3Mask = (h3 > h4)
        h4Mask = (h3 < h4)

        p = np.zeros(len(UU))
        p[h3Mask] = UU[h3Mask] ** (1. - theta[0]) * VV[h3Mask]
        p[h4Mask] = UU[h4Mask] * VV[h4Mask] ** (1. - theta[1])
        return p

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        """
        \brief H-function.
        \f[
        h(x, v) = F(x|v) = \frac{\partial C(x,v)}{\partial v}
        \f]
        """
        UU = np.asarray([u])
        VV = np.asarray([v])

        h3 = UU ** theta[0]
        h4 = VV ** theta[1]

        h3Mask = (h3 > h4)
        h4Mask = (h3 < h4)

        uu = np.zeros(len(UU))
        uu[h3Mask] = UU[h3Mask] ** (1. - theta[0])
        uu[h4Mask] = (1. - theta[1]) * UU[h4Mask] * VV[h4Mask] ** (-theta[1])
        return uu

    @CopulaBase._rotHinv
    def _hinv(self, v, u, rotation=0, *theta):
        """!
        TODO: CHECK UU and VV ordering!
        """
        U = np.asarray(u)
        V = np.asarray(v)
        uu = np.zeros(U.size)
        for i, (ui, vi) in enumerate(zip(U, V)):
            uu[i] = self._invhfun_bisect(ui, vi, rotation, *theta)
        return uu

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        raise NotImplementedError
