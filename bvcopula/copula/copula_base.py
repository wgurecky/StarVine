##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.
import numpy as np
import scipy.integrate as spi
from scipy.optimize import bisect
from scipy.optimize import minimize


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

    def _cdf(self, upper_bounds, rotation=0, *theta):
        """!
        @brief Default implementation of the cumulative density function. Very slow.
        Recommended to replace with an analytic CDF if possible.
        @param theta  Copula parameter list
        @param rotation <int> copula rotation parameter
        @param upper_bounds len=2 <np_array> [u upper, v upper]
        """
        # default implementation of bivariate CDF given _pdf()
        # copula is always supported on unit square: [0, 1]
        ranges = np.array([[0, upper_bounds[0]], [0, upper_bounds[1]]])
        return spi.nquad(self.pdf, ranges, args=(rotation, theta))[0]

    def pdf(self, u, v, rotation, theta):
        """!
        @brief Public facing PDF function.
        @param theta  <list> of <float> Copula parameter list
        @param rotation <int> copula rotation parameter
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        """
        # expand parameter list
        return self._pdf(u, v, rotation, *theta)

    def _pdf(self, u, v, rotation=0, *theta):
        """!
        @brief Pure virtual density function.
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        @param rotation_theta <int> Copula rotation (0 == 0deg, 1==90deg, ...)
        """
        return None

    def _ppf(self, u, v, rotation=0, *theta):
        """!
        @brief Percentile point function.  Equivilent to the inverse of the
        CDF.  Used to draw random samples from the bivariate distribution.
        EX:
            # will draw 100 samples from My_Copula with param0 == 0.21
            >> My_Copula._ppf(np.random.uniform(0, 1, 100), np.random.uniform(0, 1, 100), rotation=0, 0.21)
        """
        raise NotImplementedError

    def _h(self):
        """!
        @brief
        """
        raise NotImplementedError

    def _hinv(self):
        """!
        @brief Inverse H function.
        """
        pass

    def fitMLE(self, u, v, rotation=0, *theta0, **kwargs):
        """!
        @brief Maximum likelyhood copula fit.
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        @param theta0 Initial guess for copula parameter list
        """
        params0 = theta0
        res = \
            minimize(lambda args: self._nlogLike(u, v, rotation, *args),
                     params0,
                     bounds=kwargs.pop("bounds", None),
                     tol=1e-8, method='SLSQP')
        return res.x  # return best fit theta(s)

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

    def setRotation(self, rotation=0):
        self.rotation = rotation
