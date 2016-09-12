##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.
import numpy as np
import scipy.integrate as spi

class CopulaBase(object):
    def __init__(self):
        pass

    def _pairLogLike(self):
        """!
        @brief Log likelyhood function of the copula.
        Wrapps the PDF of the copula to produce the log-likelyhood
        required for MLE fitting operations.
        """
        pass

    def _cdf(self, upper_bounds, theta, rotation=0):
        """!
        @brief Default implementation of the cumulative density function. Very slow.
        Recommended to implement a analytic CDF if possible.
        @param theta  Copula parameter list
        @param rotation <int> copula rotation parameter
        @param upper_bounds len=2 <np_array> [u upper, v upper]
        """
        # default implementation of bivariate CDF given _pdf()
        ranges = np.array([[0, upper_bounds[0]], [0, upper_bounds[1]]])
        return spi.nquad(self.pdf, ranges, args=(rotation, theta))[0]

    def _pdf(self, u, v, theta, rotation=0):
        """!
        @brief Pure virtual density function.
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        @param rotation_theta <int> Copula rotation (0 == 0deg, 1==90deg, ...)
        """
        return None

    def _h(self):
        """!
        @brief
        """
        pass

    def _hinv(self):
        """!
        @brief Inverse H function.
        """
        pass

    def _nlogLike(self, u, v, rotation=0, *theta):
        """!
        @brief Default negative log likelyhood function.
        Used in MLE fitting
        """
        return -np.log(self._pdf(u, v, rotation, *theta))

    def _logLike(self, u, v, rotation=0, *theta):
        """!
        @brief Default log likelyhood func.
        """
        return np.log(self._pdf(u, v, rotation, *theta))
