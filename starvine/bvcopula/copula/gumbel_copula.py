##
# \brief Gumbel copula.
from __future__ import print_function, absolute_import, division
import numpy as np
from starvine.bvcopula.copula.copula_base import CopulaBase


class GumbelCopula(CopulaBase):
    """!
    @brief Gumbel copula
    single paramter model
    \f$\theta \in [1, \infty) \f$
    """
    def __init__(self, rotation=0, init_params=None):
        super(GumbelCopula, self).__init__(rotation, params=init_params)
        self.thetaBounds = ((1 + 1e-9, np.inf),)
        self.theta0 = (2.0, )
        self.rotation = rotation
        self.name = 'gumbel'

    @CopulaBase._rotPDF
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Probability density function for gumbel bivariate copula
        """
        h1 = theta[0] - 1.0
        # h2 = (1.0 - 2.0 ** theta[0]) / theta[0]
        h2 = (1.0 - 2.0 * theta[0]) / theta[0]
        h3 = 1.0 / theta[0]

        UU = np.asarray(u)
        VV = np.asarray(v)

        h4 = -np.log(UU)
        h5 = -np.log(VV)
        h6 = np.power(h4, theta[0]) + np.power(h5, theta[0])
        h7 = np.power(h6, h3)

        p = np.exp(-h7+h4+h5)*np.power(h4,h1)*np.power(h5,h1)*np.power(h6,h2)*(h1+h7)
        return p

    @CopulaBase._rotCDF
    def _cdf(self, u, v, rotation=0, *theta):
        h1 = 1 / theta[0]

        UU = np.asarray(u)
        VV = np.asarray(v)

        h2 = -np.log(UU)
        h3 = -np.log(VV)
        h4 = np.power(h2, theta[0]) + np.power(h3, theta[0])
        h5 = np.power(h4, h1)

        p = np.exp(-h5)
        return p

    @CopulaBase._rotH
    def _h(self, v, u, rotation=0, *theta):
        """
        TODO: CHECK UU and VV ordering!
        """
        h1 = theta[0] - 1.0
        h2 = (1.0 - theta[0]) / theta[0]
        h3 = 1.0 / theta[0]

        UU = np.asarray(1. - u)
        VV = np.asarray(1. - v)

        h4 = -np.log(VV)
        h5 = np.power(-np.log(UU), theta[0]) + np.power(h4, theta[0])

        uu = np.power(h4,h1)/VV*(np.power(h5,h2))*np.exp(-np.power(h5,h3))
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
        # Apply rotation

        # TODO: Fix! Properly vectorize hinv!
        # TODO: This is causing an issue in c-vine sampling
        # It is faster but it is incorrect.
        #if rotation == 1:
        #    U = 1. - U
        #elif rotation == 3:
        #    V = 1. - V
        #elif rotation == 0:
        #    pass
        #vv = self.vec_newton_hinv(V, U, theta[0], z_init=np.ones(len(U)))
        #if rotation == 1 or rotation == 3:
        #    return 1. - vv
        #else:
        #    return vv
        pass

    @CopulaBase._rotGen
    def _gen(self, t, *theta):
        return np.power(-np.log(t), theta[0])

    def _kTau(self, rotation=0, *theta):
        # return self._kTau(rotation, *theta)
        if self.rotation == 1 or self.rotation == 3:
            return -1. * (1. - 1. / theta[0])
        else:
            return 1. - 1. / theta[0]

    @staticmethod
    def vec_newton_hinv(u, p, theta, z_init, under_relax=0.5, eps=1e-6, iter_max=100):
        """!
        @brief Finds the zero of h() via vectorized newtons method.
            Note: For derivation of h() and h'() See:
            Dependence Modeling with Copulas. H. Joe. pp 172.
        @param u np_1darray in (0, 1)
        @param p np_1darray in (0, 1)
        @param theta float. copula shape parameter
        @param z_init np_1darray initial guess for zeros
        @param under_relax  float.
        @param eps Convergence tol
        """
        x = - np.log(u)
        h = lambda z: z + (theta - 1.) * np.log(z) - (x + (theta - 1.) * np.log(x) - np.log(p))
        h_prime = lambda z: 1. + (theta - 1.0) / z
        z = np.asarray(z_init)
        i = 0
        # z_old = np.ones(len(z)) * 1e12
        # mask = np.empty(len(z), dtype=bool)
        while i < iter_max:
            # eps_arr = z - z_old
            # mask = eps > np.abs(eps_arr)
            z -= under_relax * h(z) / h_prime(z)
            z = np.clip(z, 1e-12, 1e5)
            # z_old = deepcopy(z)
            i += 1
        y = (z ** theta - x ** theta) ** (1. / theta)
        vv = np.exp(-y)
        return vv
