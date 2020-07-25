##
# \brief Mixture of two copula
from __future__ import print_function, absolute_import, division
import numpy as np
from scipy import stats
# STARVINE IMPORTS
from starvine.bvcopula.copula.copula_base import CopulaBase
from starvine.bvcopula.copula.mvtdstpack import mvtdstpack as mvt


class MixtureCopula(CopulaBase):
    """!
    @brief Gaussian copula model
    single parameter
    \f$\theta[0] \in (-1, 1)\f$
    """
    def __init__(self, copula_a, wt_a, copula_b, wt_b):
        wt_a = wt_a / (wt_a + wt_b)
        wt_b = wt_b / (wt_a + wt_b)
        self._name = copula_a.name + '-' + copula_b.name
        self._thetaBounds = tuple(list(copula_a.thetaBounds) + list(copula_b.thetaBounds) + [(0.,1.), (0.,1.)])
        self._theta0 = tuple(list(copula_a.theta0) + list(copula_b.theta0) + [wt_a, wt_b])
        self.fittedParams = list(copula_a.theta0) + list(copula_b.theta0) + [wt_a, wt_b]
        self._copula_a = copula_a
        self._copula_b = copula_b
        self._n_params_a = len(copula_a.thetaBounds)
        self._n_params_b = len(copula_b.thetaBounds)

    @property
    def wgts(self):
        return self._fittedParams[-2:]

    @property
    def thetaBounds(self):
        return self._thetaBounds

    @property
    def theta0(self):
        return self._theta0

    @property
    def name(self):
        return self._name

    @property
    def rotation(self):
        # mixture copula has no defined rotation
        return 0

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        theta_a = theta[0:self._n_params_a]
        theta_b = theta[self._n_params_a:-2]
        wgt_a = theta[-2]
        wgt_b = theta[-1]
        return wgt_a * self._copula_a.pdf(u, v, self._copula_a.rotation, *theta_a) + \
            wgt_b * self._copula_b.pdf(u, v, self._copula_b.rotation, *theta_b)

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        theta_a = theta[0:self._n_params_a]
        theta_b = theta[self._n_params_a:-2]
        wgt_a = theta[-2]
        wgt_b = theta[-1]
        return wgt_a * self._copula_a.cdf(u, v, self._copula_a.rotation, *theta_a) + \
            wgt_b * self._copula_b.cdf(u, v, self._copula_b.rotation, *theta_b)

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        theta_a = theta[0:self._n_params_a]
        theta_b = theta[self._n_params_a:-2]
        wgt_a = theta[-2]
        wgt_b = theta[-1]
        return wgt_a * self._copula_a.h(u, v, self._copula_a.rotation, *theta_a) + \
            wgt_b * self._copula_b.h(u, v, self._copula_b.rotation, *theta_b)

    @CopulaBase._rotHinv
    def _hinv(self, v, u, rotation=0, *theta):
        theta_a = theta[0:self._n_params_a]
        theta_b = theta[self._n_params_a:-2]
        wgt_a = theta[-2]
        wgt_b = theta[-1]
        return wgt_a * self._copula_a.hinv(u, v, self._copula_a.rotation, *theta_a) + \
            wgt_b * self._copula_b.hinv(u, v, self._copula_b.rotation, *theta_b)

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        """!
        @brief Copula generating function
        """
        raise NotImplementedError
