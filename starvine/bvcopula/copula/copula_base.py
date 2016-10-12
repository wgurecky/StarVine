##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.
from __future__ import print_function, absolute_import
import numpy as np
import scipy.integrate as spi
from scipy.optimize import bisect
from scipy.optimize import minimize
from scipy.misc import derivative


class CopulaBase(object):
    """!
    @brief Bivariate Copula base class.  Meant to be subclassed
    with the PDF and CDF methods being overridden for
    a given specific copula.

    Copula can be rotated by 90, 180, 270 degrees to accommodate
    negative dependence.
    """
    def __init__(self, rotation=0):
        # Set orientation of copula
        self.rotation = rotation
        self.fittedParams = None

    # ---------------------------- PUBLIC METHODS ------------------------------ #
    def cdf(self, u, v, rotation=0, *theta):
        """!
        @brief Evaluate the copula's CDF function.
        @param u <np_1darray> Rank data vector
        @param v <np_1darray> Rank data vector
        @param rotation <int> copula rotation parameter
        @param theta  <list> of <float> Copula parameter list
        """
        return self._cdf(u, v, rotation, *theta)

    def pdf(self, u, v, rotation=0, theta=None):
        """!
        @brief Public facing PDF function.
        @param u <np_1darray> Rank data vector
        @param v <np_1darray> Rank data vector
        @param rotation <int> copula rotation parameter
        @param theta  <list> of <float> Copula parameter list
        """
        # expand parameter list
        return self._pdf(u, v, rotation, *theta)

    def h(self, u, v, rotation=0, theta=None):
        return self._h(u, v, rotation, *theta)

    def hinv(self, u, v, rotation=0, theta=None):
        return self._hinv(u, v, rotation, *theta)

    def fitMLE(self, u, v, rotation=0, *theta0, **kwargs):
        """!
        @brief Maximum likelyhood copula fit.
        @param u <np_1darray> Rank data vector
        @param v <np_1darray> Rank data vector
        @param theta0 Initial guess for copula parameter list
        @return <np_array> Array of MLE fit copula parameters
        """
        if None in theta0:
            params0 = self.theta0
        else:
            params0 = theta0
        res = \
            minimize(lambda args: self._nlogLike(u, v, rotation, *args),
                     x0=params0,
                     bounds=kwargs.pop("bounds", self.thetaBounds),
                     tol=kwargs.pop("tol", 1e-8),
                     method=kwargs.pop("method", 'SLSQP'))
        # store optimal copula params
        self.fittedParams = res.x
        return res.x  # return best fit coupula params (theta(s))

    def sample(self, n=1000, rotation=0, *theta):
        """!
        @brief Draw N samples from the copula.
        @param n Number of samples
        @param theta  Parameter list
        @param rotation Copula rotation parameter
        @return <np_array> (n, 2) size vector.  Resampled (U, V)
        data pairs from copula with paramters: *theta
        """
        u = np.random.uniform(1e-9, 1 - 1e-9, n)
        v = np.random.uniform(1e-9, 1 - 1e-9, n)
        u_hat = u
        v_hat = self._hinv(u_hat, v, rotation, *theta)
        return (u_hat, v_hat)

    def setRotation(self, rotation=0):
        """!
        @brief  Set the copula's orientation:
            0 == 0 deg
            1 == 90 deg rotation
            2 == 180 deg rotation
            3 == 270 deg rotation
        Allows for modeling negative dependence with the
        frank, gumbel, and clayton copulas (Archimedean Copula family is
        non-symmetric)
        """
        self.rotation = rotation

    # ---------------------------- PRIVATE METHODS ------------------------------ #
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Pure virtual density function.
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        @param rotation_theta <int> Copula rotation (0 == 0deg, 1==90deg, ...)
        """
        raise NotImplementedError

    def _cdf(self, u, v, rotation=0, *theta):
        """!
        @brief Default implementation of the cumulative density function. Very slow.
        Recommended to replace with an analytic CDF if possible.
        @param theta  Copula parameter list
        @param rotation <int> copula rotation parameter
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        """
        # for i, (ui, vi) in enumerate(zip(u, v)):
        #     reducedPPF = lambda pu, pv: \
        #         np.abs(np.sum(np.array([ui, vi]) - self._ppf(pu, pv, 0, *theta)))
        #     cdf_vector[i] = fsolve(reducedPPF, x0=np.array([0.5, 0.5]))[0]
        # return cdf_vector
        cdf_vector = np.zeros(np.array(u).size)
        for i, (ui, vi) in enumerate(zip(u, v)):
            ranges = np.array([[0, ui], [0, vi]])
            cdf_vector[i] = spi.nquad(self.pdf, ranges,
                                      args=(rotation, theta),
                                      opts={'limit': 20})[0]
        return cdf_vector

    def _ppf(self, u, v, rotation=0, *theta):
        """!
        @brief Percentile point function.  Equivilent to the inverse of the
        CDF.  Used to draw random samples from the bivariate distribution.

        EX: will draw 100 samples from a t-copula with params [0.21, 20]
        >>> import starvine.bvcopula as bvc
        >>> My_Copula = bvc.t_copula.StudentTCopula()
        >>> u, v = np.random.uniform(0, 1, 100)
        >>> My_Copula._ppf(u, v, rotation=0, 0.21, 20)
        """
        u_hat = u
        v_hat = self._hinv(u_hat, v, rotation, *theta)
        return (u_hat, v_hat)

    def _h(self, u, v, rotation=0, *theta):
        """!
        @brief Copula conditional distribution function.
        Provides \f$ h \f$  given \f$ V \f$  and \f$ \theta \f$
        \f$ h(u|v, \theta) = \frac{\partial C( F(u|v), F(u|v) | \theta) }{\partial F(u|v)} \f$

        @param u <np_1darray> is uniformly distributed on [0, 1]
        @param v <np_1darray> is distributed acording to some some known DF
        @param rotation Copula rotation paramter
        @param theta  Known copula paramter list
        @return h \f$ h(u|v, \theta) \f$
        """
        raise NotImplementedError

    def _hinv(self, u, v, rotation=0, *theta):
        """!
        @brief Inverse H function.
        """
        raise NotImplementedError

    def _v(self, u, v, rotation=0, *theta):
        """!
        @brief Copula conditional distribution function.
        Provides \f$ V\f$  given \f$ u \f$  and \f$ \theta \f$
        """
        # For symmetric copula: flip args
        self._h(v, u, rotation=0, *theta)

    def _vinv(self, u, v, rotation=0, *theta):
        # For symmetric copula: flip args
        self._hinv(v, u, rotation=0, *theta)

    def _nlogLike(self, u, v, rotation=0, *theta):
        """!
        @brief Default negative log likelyhood function.
        Used in MLE fitting
        """
        return -self._logLike(u, v, rotation, *theta)

    def _logLike(self, u, v, rotation=0, *theta):
        """!
        @brief Default log likelyhood func.
        """
        return np.sum(np.log(self._pdf(u, v, rotation, *theta)))

    def _invhfun_bisect(self, U, V, rotation, *theta):
        """!
        @brief Compute inverse of H function using bisection.
        TODO: Improve performance: finish with newton iterations
        """
        # Freeze U, V, rotation, and model parameter, theta
        reducedHfn = lambda u: self._h(u, V, rotation, *theta) - U
        return bisect(reducedHfn, 1e-10, 1.0 - 1e-10, maxiter=500)[0]

    def _AIC(self, u, v, rotation=0, *theta):
        """!
        @brief Estimate the AIC of a fitted copula (with params == theta)
        @param theta Copula paramter list
        """
        cll = self._nlogLike(u, v, rotation, *theta)
        if len(theta) == 1:
            # 1 parameter copula
            AIC = 2 * cll + 2.0 + 4.0 / (len(u) - 2)
        else:
            # 2 parameter copula
            AIC = 2 * cll + 4.0 + 12.0 / (len(u) - 3)
        return AIC

    def _gen(self, t, *theta):
        """!
        @brief Copula generator function.
        """
        raise NotImplementedError

    def kTau(self, rotation=0, *theta):
        """!
        @brief Public facing kendall's tau function.
        """
        raise NotImplementedError

    def _kTau(self, rotation=0, *theta):
        """!
        @brief Computes Kendall's tau.  Requires that
        the copula has a _gen() method implemented.
        This method should be overridden if an analytic form of
        kendall's tau is avalible.

        Let \f$T = C(u, v)$\f represent a univariate random variable
        which is in turn a function of the random variables, u \& v.
        \f$ K_c(t) = t - \frac{\phi(t)}{\phi'(t)} $\f
        where \f$ \phi(t) $\f is the copula generating function.

        Note:
            For the gauss and T copula this should be == (2.0/np.pi) * arcsin(rho)
            where rho is the T and gauss correlation parameter.
        """
        t_range = np.array([[1e-9, 1 - 1e-9], ])
        # Kendall's tau distribution
        K_c = lambda t: t - self._gen(t, *theta) / \
            derivative(self._gen, t, dx=1e-5, args=(rotation, theta))
        # Cumulative copula
        cumCopula = spi.nquad(K_c, t_range, args=(rotation, theta))[0]
        return 3.0 - 4.0 * cumCopula

    # -------------------------- COPULA ROTATION METHODS ---------------------------- #
    @classmethod
    def _rotPDF(cls, f):
        """!
        @brief Define copula probability density function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v = args[0], args[1]
            nargs = args[2:]
            if not nargs:
                nargs = self.fittedParams
                # TODO: raise  error if Not fittedParams and no *args
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, *args, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return f(self, 1. - u, v, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return f(self, 1. - u, 1. - v, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                return f(self, u, 1. - v, *nargs)
        return wrapper

    @classmethod
    def _rotCDF(cls, f):
        """!
        @brief Define copula cumulative density function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v = args[0], args[1]
            nargs = args[2:]
            if not nargs:
                nargs = self.fittedParams
                # TODO: raise  error if Not fittedParams and no *args
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, *args, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return v - f(self, 1. - u, v, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return f(self, 1. - u, 1. - v, *nargs) + u + v - 1.
            elif self.rotation == 3:
                # 270 deg rotation
                return u - f(self, u, 1. - v, *nargs)
        return wrapper

    @classmethod
    def _rotHinv(cls, f):
        """!
        @brief Define copula dependence function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v = args[0], args[1]
            nargs = args[2:]
            if not nargs:
                nargs = self.fittedParams
                # TODO: raise  error if Not fittedParams and no *args
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, *args, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return 1. - f(self, 1. - u, v, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return 1. - f(self, 1. - u, 1. - v, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                return f(self, u, 1. - v, *nargs)
        return wrapper

    @classmethod
    def _rotH(cls, f):
        """!
        @brief Define copula inverse dependence function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v = args[0], args[1]
            nargs = args[2:]
            if not nargs:
                nargs = self.fittedParams
                # TODO: raise  error if Not fittedParams and no *args
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, *args, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return 1. - f(self, 1. - u, v, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return 1. - f(self, 1. - u, 1. - v, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                return f(self, u, 1. - v, *nargs)
        return wrapper
