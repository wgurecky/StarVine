##
# @brief Lognormal model distribution
from __future__ import print_function, division
from uv_base import UVmodel
import numpy as np
from scipy.special import erf


class UVLogNorm(UVmodel):
    """!
    @brief Custom 2 parameter gaussian distribution.

    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # LogNormal PDF is supported on (0, +\infty)
        # takes (mean, std. dev) as paramters
        super(UVLogNorm, self).__init__(paramsString="m, s",
                                        momtype=0,
                                        bounds=[1e-9, np.inf],
                                        name=kwargs.pop("name", "custom_lognorm"))
        self.defaultParams = [2.0, 1.]

    def _pdf(self, x, *args):
        """!
        @brief Lognorm PDF
        """
        m, s, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return (1.0 / (x * np.sqrt(2.0 * np.pi * s))) * \
            np.exp(-(x - m) ** 2 / (2 * s))

    def _cdf(self, x, *args):
        """!
        @brief Lognorm CDF
        """
        m, s, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return 0.5 * (1 + erf((np.log(x) - m) / (np.sqrt(s) * np.sqrt(2))))

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        if params[1] <= 0.001 or params[0] <= 0:
            return False
        return True
