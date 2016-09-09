##
# \brief Bivariate copula base class.
# All copula must have a density function, CDF, and
# H function.


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

    def _cdf(self):
        """!
        @brief internal implementation of the cumulative density function.
        """
        pass

    def _pdf(self, u, v, rotation_theta=0):
        """!
        @brief internal implementation of the density function.
        @param u <np_1darray> Rank CDF data vector
        @param v <np_1darray> Rank CDF data vector
        @param rotation_theta <int> Copula rotation (0 == 0deg, 1==90deg, ...)
        """
        pass

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
