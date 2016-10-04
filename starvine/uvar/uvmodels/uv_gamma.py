##
# @brief Gamma model distribution
from __future__ import print_function, division
from uv_base import UVmodel
import numpy as np
from scipy.special import gamma


class UVGamma(UVmodel):
    """!
    @brief Custom 2 parameter gamma distribution.

    Note:
        Included example.
    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Gamma PDF is supported on (0, +\infty)
        super(UVGamma, self).__init__(paramsString="a, b",
                                      momtype=0,
                                      bounds=[0, np.inf],
                                      name=kwargs.pop("name", "custom_gamma"))
        self.defaultParams = [9.0, 2.]

    def _pdf(self, x, *args):
        """!
        @brief Gamma PDF
        """
        a, b, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return (b ** a) * (x ** (a - 1.0)) * np.exp(-x * b) / gamma(a)

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        for p in params:
            if p <= 0:
                return False
        return True
