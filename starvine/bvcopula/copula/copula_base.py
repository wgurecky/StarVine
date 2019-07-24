##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.
from __future__ import print_function, absolute_import, division
import numpy as np
import six, abc
import scipy.integrate as spi
from scipy.optimize import bisect, newton, brentq
from scipy.optimize import minimize
from scipy.misc import derivative
import warnings
warnings.filterwarnings('ignore')


@six.add_metaclass(abc.ABCMeta)
class CopulaBase(object):
    """!
    @brief Bivariate Copula base class.  Meant to be subclassed
    with the PDF and CDF methods being overridden for
    a given specific copula.

    Copula can be rotated by 90, 180, 270 degrees to accommodate
    negative dependence.
    """
    def __init__(self, rotation=0, thetaBounds=((-np.inf, np.inf),),
                 theta0=(0.0,), name='defaut', **kwargs):
        """!
        @brief Init copula
        @param rotation <b>int</b>  Copula orientation in [0, 1, 2, 3]
        @param thetaBounds <b>tuple of tuples</b> should have len = len(theta0).
            provides upper and lower bounds for each copula parameter
        @param theta0 <b>tuple</b> initial parameter guesses
        @name <b>str</b> name of coupula
        """
        self._thetaBounds = thetaBounds
        self.rotation = rotation
        self.theta0 = theta0
        self.name = name
        self._fittedParams = kwargs.pop("params", None)

    @property
    def fittedParams(self):
        """!
        @brief Fitted parameters from MLE or MCMC
        """
        return self._fittedParams

    @fittedParams.setter
    def fittedParams(self, fp):
        if not self._bounds_check(*fp):
            raise RuntimeError("Fitted params not in bounds.")
        self._fittedParams = fp

    @property
    def theta0(self):
        """!
        @brief Initial parameter guess
        """
        return self._theta0

    @theta0.setter
    def theta0(self, theta):
        if not self._bounds_check(*theta):
            raise RuntimeError("Theta_0 not in bounds.")
        self._theta0 = theta

    @property
    def thetaBounds(self):
        """!
        @brief Copula parameter bounds
        """
        return self._thetaBounds

    @thetaBounds.setter
    def thetaBounds(self, bounds):
        """!
        @brief Copula parameter bounds
        """
        self._thetaBounds = bounds

    @property
    def rotation(self):
        """!
        @brief Copula orientation
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rot):
        assert rot in [0, 1, 2, 3]
        self._rotation = rot

    @property
    def name(self):
        """!
        @brief Copula name
        """
        return self._name

    @name.setter
    def name(self, nm_str):
        self._name = nm_str

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

    def fitMcmc(self, u, v, *theta0, **kwargs):
        """!
        @brief Markov chain monte carlo fit method
        @param u <b>np_1darray</b> Rank data vector
        @param v <b>np_1darray</b> Rank data vector
        @param theta0 Initial guess for copula parameter list
        @return <b>tuple</b> :
                (<b>np_array</b> Array of MLE fit copula parameters,
                <b>np_2darray</b> sample array of shape (nparams, nsamples))
        """
        from emcee import EnsembleSampler
        wgts = kwargs.pop("weights", np.ones(len(u)))
        rotation = 0
        ln_prob = lambda theta: self._ln_prior(*theta, **kwargs) + \
                self._ln_like(u, v, wgts, rotation, *theta)
        if None in theta0:
            params0 = self.theta0
        else:
            params0 = theta0
        ndim = len(params0)
        ngen = kwargs.get("ngen", 200)
        nburn = kwargs.get("nburn", 100)
        nwalkers = kwargs.get("nwalkers", 50)
        # initilize walkers in gaussian ball around theta0
        pos_0 = [np.array(params0) + 1e-6 * np.asarray(params0)*np.random.randn(ndim) for i in range(nwalkers)]
        emcee_mcmc = EnsembleSampler(nwalkers, ndim, ln_prob)
        emcee_mcmc.run_mcmc(pos_0, ngen)
        samples = emcee_mcmc.chain[:, nburn:, :].reshape((-1, ndim))
        res = np.mean(samples, axis=0)
        self._fittedParams = res
        return res, samples

    def fitMLE(self, u, v, *theta0, **kwargs):
        """!
        @brief Maximum likelihood copula fit.
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
                             bounds=kwargs.pop("bounds", self.thetaBounds),
                             )
            else:
                res = \
                    minimize(lambda args: self._nlogLike(u, v, wgts, rotation, *args),
                             x0=params0,
                             bounds=kwargs.pop("bounds", self.thetaBounds),
                             tol=kwargs.pop("tol", 1e-8),
                             method=kwargs.pop("altMethod", 'L-BFGS-B'))
        if not res.success:
            print("WARNING: Copula parameter fitting failed to converge!")
        self._fittedParams = res.x
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

    def sampleScale(self, frozen_margin_x, frozen_margin_y, n, *mytheta):
        """!
        @brief Draw N samples from the bivariate copula and scale the
        results according to input model cdfs.
        @param x  1d_vector of abscissa for component 1
        @param y  1d_vector of abscissa for component 2
        @param frozen_margin_x marginal distribution function for component 1
        @param frozen_margin_y marginal distribution function for component 2
        @param mytheta  (optional) Copula parameter list
        @return scaled samples from the bivariate copula model.
        """
        u_hat, v_hat = self.sample(n, *mytheta)
        resampled_scaled_x = self.icdf_uv_bisect(u_hat, frozen_margin_x)
        resampled_scaled_y = self.icdf_uv_bisect(v_hat, frozen_margin_y)
        return (resampled_scaled_x, resampled_scaled_y)

    def setRotation(self, rotation=0):
        """!
        @brief  Set the copula's orientation:
        Allows for modeling negative dependence with the
        frank, gumbel, and clayton copulas
        @param rotation <b>int</b> Copula rotation.
            0 == 0 deg,
            1 == 90 deg rotation,
            2 == 180 deg rotation,
            3 == 270 deg rotation
        """
        self.rotation = rotation

    # ---------------------------- PRIVATE METHODS ------------------------------ #
    @abc.abstractmethod
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
        cdf_vector = np.zeros(np.asarray(u).size)
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        @brief Default negative log likelihood function.
        Used in MLE fitting
        """
        return -1.0 * self._logLike(u, v, wgts, rotation, *theta)

    def _logLike(self, u, v, wgts=None, rotation=0, *theta):
        """!
        @brief Default log likelihood func.
        """
        if wgts is None:
            wgts = np.ones(len(u))
        return np.sum(wgts * np.log(self._pdf(u, v, rotation, *theta)))

    def _ln_like(self, u, v, wgts=None, rotation=0, *theta):
        """!
        @brief log likelihood func wrapper.  return -np.inf if
        theta is out of bounds
        """
        try:
            a = self._logLike(u, v, wgts, rotation, *theta)
            if np.isnan(a):
                return -np.inf
            else:
                return a
        except:
            return -np.inf

    def _bounds_check(self, *theta, **kwargs):
        """!
        @brief Check if parameters are in bounds.
        """
        bounds=kwargs.pop("bounds", self.thetaBounds)
        assert len(theta) == len(bounds)
        for i, param in enumerate(theta):
            if bounds[i][0] < param < bounds[i][1]:
                pass
            else:
                return False
        return True

    def _ln_prior(self, *theta, **kwargs):
        in_bounds = self._bounds_check(*theta, **kwargs)
        if in_bounds:
            return 0.0
        else:
            return -np.inf

    def _invhfun_bisect(self, U, V, rotation, *theta):
        """!
        @brief Compute inverse of H function using bisection.
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
        try:
            v_est_ = brentq(reducedHfn, a=1e-60, b=1.0-1e-60, maxiter=18, xtol=1e-10, rtol=1e-12)
        except:
            # fallback if newton fails to converge
            print("WARNING: hinv root find failed. Falling back to bisection.")
            v_est_ = bisect(reducedHfn, 1e-60, 1.0 - 1e-60, maxiter=50, disp=False)
        # return v_est_
        v_est_ = np.clip(v_est_, 1e-8, 1-1e-8)
        if self.rotation == 1 or self.rotation == 0:
            return 1. - v_est_
        else:
            return v_est_

    def _AIC(self, u, v, rotation=0, *theta, **kwargs):
        """!
        @brief Estimate the AIC of a fitted copula (with params == theta)
        @param u  np_1darray. random variable samples uniform distributed on [0, 1]
        @param v  np_1darray. random variable samples uniform distributed on [0, 1]
        @param theta Copula paramter list
        """
        wgts = kwargs.pop("weights", np.ones(len(u)))
        cll = self._nlogLike(u, v, wgts, rotation, *theta)
        k = len(self.theta0)
        AIC = 2 * cll + 2.0 * k
        AICc = AIC + (2. * k ** 2. + 2. * k) / (len(u) - k - 1)
        return AICc

    @abc.abstractmethod
    def _gen(self, t, *theta):
        """!
        @brief Copula generator function.
        """
        raise NotImplementedError

    def fitKtau(self, kTau, **kwargs):
        """!
        @brief Given kTau, estimate the copula parameter.
        Only avalible for single parameter copula
        models.
        Solves the minimization problem:
        \f[
        argmin_\theta (\tau - \hat\tau(\theta))
        \f]
        @param kTau  float. Specified kendall's tau.
        """
        if len(self.theta0) != 1:
            raise RuntimeError("ERROR: kendall's tau fit only possible with single parameter copula.")
        # initial guess
        if not self._fittedParams:
            param = self.theta0[0]
        else:
            param = self._fittedParams[0]
        # obj func: kendall's tau square err as function of param
        ktf = lambda p: (self.kTau(self.rotation, p) - kTau) ** 2.
        res = \
            minimize(ktf, x0=param,
                     bounds=kwargs.pop("bounds", self.thetaBounds),
                     tol=kwargs.pop("tol", 1e-8),
                     method=kwargs.pop("method", 'BFGS'))
        if not res.success:
            res = \
                minimize(ktf, x0=param,
                         bounds=kwargs.pop("bounds", self.thetaBounds),
                         tol=kwargs.pop("tol", 1e-8),
                         method=kwargs.pop("method", 'Nelder-Mead'))

        if not res.success:
            print("WARNING: Copula parameter fitting failed to converge!")
        self._fittedParams = res.x
        return res.x, res.success

    def kTau(self, rotation=0, *theta):
        """!
        @brief Computes kendall's tau.
        @param rotation Optional copula rotation parameter.  If
            unspecified, automatically determined by self.rotation setting.
        """
        if not any(theta):
            theta = self._fittedParams
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
        This function is valid for Archimedean copula ONLY.

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

    def kC(self, t_in, *theta):
        """!
        @brief evaluates kendall's function.
        The kendall's function is the copula equivillent to a
        multivariate probability integral transform:
        \f[
           K(t|\theta) = Pr\{C(u, v| \theta) \leq t ) \}
        \f]
        The double integral can be computed numerically:
        \f[
            \int_0^t \int_0^t I{C(u, v) \leq t} dC(u, v)}
        \f]
        Where $I$ is the indicator function.
        Numerical integration is necissary in the base class as to
        not loose generality (Archimedean copula are a special case
        and kC(t) can be computed easily for these copula).
        @param t  np_1darray of evaluation points in [0, 1]
        @param rotation int. copula rotation
        """
        assert(min(t_in) >= 0)
        assert(max(t_in) <= 1)
        try:
            # Only Archimedean copula should have _gen()
            reduced_gen = lambda t: self._gen(t, *theta)

            def K_c(t):
                return reduced_gen(t) / \
                    derivative(reduced_gen, t, dx=1e-9)
            return t_in - K_c(t_in)
        except:
            kc_out = []
            # create u, v grid
            u = np.random.uniform(0, 1, int(2e4))
            v = np.random.uniform(0, 1, int(2e4))
            u_hat, v_hat = self._ppf(u, v, self.rotation, *theta)
            cdf_int = self._cdf(u_hat, v_hat, self.rotation, *theta)
            for t in t_in:
                # create mask
                ones_mask = (cdf_int <= t)
                # compute probability
                kc_out.append(float(np.count_nonzero(ones_mask) / len(ones_mask)))
            return np.array(kc_out)

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
                nargs = self._fittedParams
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
                nargs = self._fittedParams
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
                nargs = self._fittedParams
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
                nargs = self._fittedParams
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
    def _rotGen(cls, f):
        """!
        @brief Copula generator wrapper
        """
        def wrapper(self, *args, **kwargs):
            t = args[0]
            nargs = args[1:]
            if not any(nargs):
                nargs = self._fittedParams
            if not any(nargs):
                raise RuntimeError("Parameter missing")
            return f(self, t, *nargs)
        return wrapper

    @staticmethod
    def icdf_uv_bisect(X, frozen_marginal_model):
        """
        @brief Apply marginal model.
        @param X <b>np_1darray</b> samples from copula margin (uniform distributed)
        @param frozen_marginal_model frozen scipy.stats.rv_continuous python object
            or <b>function</b> inverse cdf fuction
            Note: inv_CDF should be a monotonic function with a domain in [0, 1]
        @return <b>np_1darray</b> Samples drawn from supplied margin model
        """
        if hasattr(frozen_marginal_model, 'ppf'):
            icdf = frozen_marginal_model.ppf(X)
        else:
            icdf = frozen_marginal_model(X)
        return icdf
