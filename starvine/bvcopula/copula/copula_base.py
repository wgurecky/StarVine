##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.
from __future__ import print_function, absolute_import
import numpy as np
import scipy.integrate as spi
from scipy.optimize import bisect, newton
from scipy.optimize import minimize
from scipy.misc import derivative
import warnings
warnings.filterwarnings('ignore')


class CopulaBase(object):
    """!
    @brief Bivariate Copula base class.  Meant to be subclassed
    with the PDF and CDF methods being overridden for
    a given specific copula.

    Copula can be rotated by 90, 180, 270 degrees to accommodate
    negative dependence.
    """
    def __init__(self, rotation=0):
        """!
        @brief Init copula
        @param rotation <b>int</b>  Copula orientation
        """
        self.rotation = rotation
        self.fittedParams = None

    # ---------------------------- PUBLIC METHODS ------------------------------ #
    def cdf(self, u, v, *theta):
        """!
        @brief Evaluate the copula's CDF function.
        @param u <b>np_1darray</b> Rank data vector
        @param v <b>np_1darray</b> Rank data vector
        @param theta  <b>list</b> of <b>float</b> Copula parameter list
        """
        rotation = 0
        return self._cdf(u, v, rotation, *theta)

    def pdf(self, u, v, *theta):
        """!
        @brief Public facing PDF function.
        @param u <b>np_1darray</b> Rank data vector
        @param v <b>np_1darray</b> Rank data vector
        @param theta  <b>list</b> of <b>float</b> Copula parameter list
        """
        rotation = 0
        return self._pdf(u, v, rotation, *theta)

    def h(self, u, v, *theta):
        rotation = 0
        return self._h(u, v, rotation, *theta)

    def hinv(self, u, v, *theta):
        rotation = 0
        return self._hinv(u, v, rotation, *theta)

    def fitMLE(self, u, v, *theta0, **kwargs):
        """!
        @brief Maximum likelyhood copula fit.
        @param u <b>np_1darray</b> Rank data vector
        @param v <b>np_1darray</b> Rank data vector
        @param theta0 Initial guess for copula parameter list
        @return <b>tuple</b> :
                (<b>np_array</b> Array of MLE fit copula parameters,
                <b>int</b> Fitting success flag, 1==success)
        """
        wgts = kwargs.pop("weights", np.ones(len(u)))
        rotation = 0
        if None in theta0:
            params0 = self.theta0
        else:
            params0 = theta0
        res = \
            minimize(lambda args: self._nlogLike(u, v, wgts, rotation, *args),
                     x0=params0,
                     bounds=kwargs.pop("bounds", self.thetaBounds),
                     tol=kwargs.pop("tol", 1e-8),
                     method=kwargs.pop("method", 'SLSQP'))
        if not res.success:
            # Fallback
            if "frank" in self.name:
                res = \
                    minimize(lambda args: self._nlogLike(u, v, wgts, rotation, *args),
                             x0=params0,
                             tol=kwargs.pop("tol", 1e-8),
                             method=kwargs.pop("altMethod", 'Nelder-Mead'))
            else:
                res = \
                    minimize(lambda args: self._nlogLike(u, v, wgts, rotation, *args),
                             x0=params0,
                             bounds=kwargs.pop("bounds", self.thetaBounds),
                             tol=kwargs.pop("tol", 1e-8),
                             method=kwargs.pop("altMethod", 'L-BFGS-B'))
        if not res.success:
            print("WARNING: Copula parameter fitting failed to converge!")
        self.fittedParams = res.x
        return res.x, res.success  # return best fit coupula params (theta(s))

    def sample(self, n=1000, *mytheta):
        """!
        @brief Draw N samples from the copula.
        @param n Number of samples
        @param mytheta  Parameter list
        @return <b>np_array</b> (n, 2) size vector of samples from bivariate copula model.
        """
        rotation = 0
        u_iid_uniform = np.random.uniform(1e-9, 1 - 1e-9, n)
        v_iid_uniform = np.random.uniform(1e-9, 1 - 1e-9, n)
        # sample from copula
        u_hat = u_iid_uniform
        v_hat = self._hinv(u_iid_uniform, v_iid_uniform, rotation, *mytheta)
        return (u_hat, v_hat)

    def sampleScale(self, x, y, xCDF, yCDF, *mytheta):
        """!
        @brief Draw N samples from the bivariate copula and scale the
        results according to input model cdfs.
        @param x  1d_vector of abscissa for component 1
        @param y  1d_vector of abscissa for component 2
        @param xCDF cumulative marginal distribution function for component 1
        @param yCDF cumulative marginal distribution function for component 2
        @param mytheta  (optional) Copula parameter list
        @return scaled samples from the bivariate copula model.
        """
        n = len(x)
        u_hat, v_hat = self.sample(n, *mytheta)
        resampled_scaled_x = icdf_uv_bisect(x, u_hat, xCDF)
        resampled_scaled_y = icdf_uv_bisect(y, v_hat, yCDF)
        return (resampled_scaled_x, resampled_scaled_y)

    def setRotation(self, rotation=0):
        """!
        @brief  Set the copula's orientation:
        Allows for modeling negative dependence with the
        frank, gumbel, and clayton copulas (Archimedean Copula family is
        non-symmetric)
        @param rotation <b>int</b> Copula rotation.
            0 == 0 deg,
            1 == 90 deg rotation,
            2 == 180 deg rotation,
            3 == 270 deg rotation
        """
        self.rotation = rotation

    # ---------------------------- PRIVATE METHODS ------------------------------ #
    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Pure virtual density function.
        @param u <b>np_1darray</b> Rank CDF data vector
        @param v <b>np_1darray</b> Rank CDF data vector
        @param rotation_theta <b>int</b> Copula rotation (0 == 0deg, 1==90deg, ...)
        """
        raise NotImplementedError

    def _cdf(self, u, v, rotation=0, *theta):
        """!
        @brief Default implementation of the cumulative density function. Very slow.
        Recommended to replace with an analytic CDF if possible.
        @param theta  Copula parameter list
        @param rotation <b>int</b> copula rotation parameter
        @param u <b>np_1darray</b> Rank CDF data vector
        @param v <b>np_1darray</b> Rank CDF data vector
        """
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
        Provides \f$ h \f$  given \f$ V \f$  and \f$ \theta \f$ .
        \f[
        h(u|v, \theta) = \frac{\partial C( F(u|v), F(u|v) | \theta) }{\partial F(u|v)}
        \f]

        @param u <b>np_1darray</b> is uniformly distributed on [0, 1]
        @param v <b>np_1darray</b> is distributed acording to some some known DF
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

    def _nlogLike(self, u, v, wgts=None, rotation=0, *theta):
        """!
        @brief Default negative log likelyhood function.
        Used in MLE fitting
        """
        if wgts is None:
            wgts = np.ones(len(u))
        return -self._logLike(u, v, wgts, rotation, *theta)

    def _logLike(self, u, v, wgts=None, rotation=0, *theta):
        """!
        @brief Default log likelyhood func.
        """
        if wgts is None:
            wgts = np.ones(len(u))
        return np.sum(wgts * np.log(self._pdf(u, v, rotation, *theta)))

    def _invhfun_bisect(self, U, V, rotation, *theta):
        """!
        @brief Compute inverse of H function using bisection.
        Update (11/08/2016) performance improvement: finish with newton iterations
        Update (12/04/2016) Newton can minimization needs bounds checks!
        TODO: move to scipy.minmize
        """
        # Apply limiters
        U = np.clip(U, 1e-8, 1. - 1e-8)
        V = np.clip(V, 1e-8, 1. - 1e-8)
        # Apply rotation
        if self.rotation == 1:
            U = 1. - U
        elif self.rotation == 3:
            V = 1. - V
        elif self.rotation == 0:
            U = 1. - U
            V = 1. - V
        else:
            pass
        reducedHfn = lambda u: self._h(V, u, rotation, *theta) - U
        v_bisect_est_ = bisect(reducedHfn, 1e-200, 1.0 - 1e-200, maxiter=20, disp=False)
        try:
            v_est_ = newton(reducedHfn, v_bisect_est_, tol=1e-5, maxiter=30)
        except:
            # fallback if newton fails to converge
            v_est_ = bisect(reducedHfn, 1e-60, 1.0 - 1e-60, maxiter=50, disp=False)
        # return v_est_
        if v_est_ > 1.0:
            return 1.0 - 1e-8
        elif v_est_ < 0.0:
            return 1e-8
        if self.rotation == 1 or self.rotation == 0:
            return 1. - v_est_
        else:
            return v_est_

    def _AIC(self, u, v, rotation=0, *theta):
        """!
        @brief Estimate the AIC of a fitted copula (with params == theta)
        @param theta Copula paramter list
        """
        cll = self._nlogLike(u, v, None, rotation, *theta)
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
        if not any(theta):
            theta = self.fittedParams
        return self._kTau(rotation, *theta)

    def _kTau(self, rotation=0, *theta):
        """!
        @brief Computes Kendall's tau.  Requires that the copula has a _gen()
        method implemented. This method should be overridden if an analytic form of
        kendall's tau is avalible.

        Let \f$ T = C(u, v) \f$ represent a random variable and
        \f$ t \f$ is an RV distributed according to \f$ T() \f$.
        \f[ K_c(t) = \frac{\phi(t)}{\phi'(t)} \f]
        \f[ \tau = 1 + 4 \int_0^1 K_c(t) dt \f]
        where \f$ \phi(t) \f$ is the copula generating function.

        Note:
        For the gauss and students-t copula:
        \f[ \tau = \frac{2.0}{\pi}  arcsin(\rho) \f]
        where \f$ \rho \f$ is the linear correlation coefficient.
        """
        t_range = np.array([[1e-8, 1 - 1e-8], ])
        reduced_gen = lambda t: self._gen(t, *theta)
        def K_c(t):
            return reduced_gen(t) / \
                derivative(reduced_gen, t, dx=1e-9)
        cumCopula = spi.nquad(K_c, t_range)[0]
        negC = 1.
        if self.rotation == 1 or self.rotation == 3:
            negC = -1.
        return negC * (1.0 + 4.0 * cumCopula)

    # -------------------------- COPULA ROTATION METHODS ---------------------------- #
    @classmethod
    def _rotPDF(cls, f):
        """!
        @brief Define copula probability density function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v, rot = args[0], args[1], args[2]
            nargs = args[3:]
            if not any(nargs):
                nargs = self.fittedParams
            if not any(nargs):
                # Raise error if fittedParams not set
                raise RuntimeError("Parameter missing")
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, u, v, rot, *nargs, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return f(self, 1. - u, v, rot, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return f(self, 1. - u, 1. - v, rot, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                return f(self, u, 1. - v, rot, *nargs)
        return wrapper

    @classmethod
    def _rotCDF(cls, f):
        """!
        @brief Define copula cumulative density function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v, rot = args[0], args[1], args[2]
            nargs = args[3:]
            if not any(nargs):
                nargs = self.fittedParams
            if not any(nargs):
                raise RuntimeError("Parameter missing")
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, u, v, rot, *nargs, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return v - f(self, 1. - u, v, rot, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return f(self, 1. - u, 1. - v, rot, *nargs) + u + v - 1.
            elif self.rotation == 3:
                # 270 deg rotation
                return u - f(self, u, 1. - v, rot, *nargs)
        return wrapper

    @classmethod
    def _rotHinv(cls, f):
        """!
        @brief Define copula dependence function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v, rot = args[0], args[1], args[2]
            nargs = args[3:]
            if not any(nargs):
                nargs = self.fittedParams
            if not any(nargs):
                raise RuntimeError("Parameter missing")
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, u, v, rot, *nargs, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return f(self, 1. - u, v, rot, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return 1. - f(self, 1. - u, 1. - v, rot, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                return 1. - f(self, u, 1. - v, rot, *nargs)
        return wrapper

    @classmethod
    def _rotH(cls, f):
        """!
        @brief Define copula inverse dependence function rotation.
        """
        def wrapper(self, *args, **kwargs):
            u, v, rot = args[0], args[1], args[2]
            nargs = args[3:]
            if not any(nargs):
                nargs = self.fittedParams
            if not any(nargs):
                raise RuntimeError("Parameter missing")
            if self.rotation == 0:
                # 0 deg rotation
                return f(self, u, v, rot, *nargs, **kwargs)
            elif self.rotation == 1:
                # 90 deg rotation
                return f(self, 1. - u, v, rot, *nargs)
            elif self.rotation == 2:
                # 180 deg rotation
                return 1. - f(self, 1. - u, 1. - v, rot, *nargs)
            elif self.rotation == 3:
                # 270 deg rotation
                # Clayton is flipped!
                # TODO: Fix GUMBEL 270 rotation
                # return 1. - f(self, 1. - u, v, rot, *nargs)  # works for gumbel
                return 1. - f(self, u, 1. - v, rot, *nargs)
        return wrapper

    @classmethod
    def _rotGen(cls, f):
        """!
        @brief Copula generator wrapper
        """
        def wrapper(self, *args, **kwargs):
            t = args[0]
            nargs = args[1:]
            if not any(nargs):
                nargs = self.fittedParams
            if not any(nargs):
                raise RuntimeError("Parameter missing")
            return f(self, t, *nargs)
        return wrapper


def icdf_uv_bisect(ux, X, marginalCDFModel):
    """
    @brief Apply marginal model.
    """
    icdf = np.zeros(np.array(X).size)
    for i, xx in enumerate(X):
        kde_cdf_err = lambda m: xx - marginalCDFModel(m)
        try:
            icdf[i] = bisect(kde_cdf_err,
                             min(ux) - np.abs(0.8 * min(ux)),
                             max(ux) + np.abs(0.8 * max(ux)),
                             xtol=1e-2, maxiter=25)
            icdf[i] = newton(kde_cdf_err, icdf[i], tol=1e-6, maxiter=10)
        except:
            # icdf[i] = np.nan
            pass
    return icdf
