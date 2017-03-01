##
# @brief Beta model distribution
from __future__ import print_function, division
from uv_base import UVmodel
import numpy as np
from scipy.special import gamma


class UVExp(UVmodel):
    """!
    @brief Custom 1 parameter exponential distribution.
    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Beta PDF is supported on (0, 1)
        super(UVExp, self).__init__(paramsString="a",
                                     momtype=0,
                                     bounds=[0.0, np.inf],
                                     name=kwargs.pop("name", "custom_exp"))
        self.defaultParams = [1.5, 5.]

    def _pdf(self, x, *args):
        """!
        @brief Exponential PDF
        """
        a, x = np.ravel(args[0]), np.ravel(x)
        return a * np.exp(-a * x)

    def _cdf(self, x, *args):
        a, x = np.ravel(args[0]), np.ravel(x)
        return 1. - np.exp(-a * x)

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        for p in params:
            # all params >0
            if p <= 1e-6:
                return False
        return True
