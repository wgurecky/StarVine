##
# @brief Gaussian model distribution
from __future__ import print_function, division
from uv_base import UVmodel
import numpy as np


class UVGauss(UVmodel):
    """!
    @brief Custom 2 parameter gaussian distribution.

    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Gamma PDF is supported on (0, +\infty)
        super(UVGauss, self).__init__(paramsString="m, s",
                                      momtype=0,
                                      bounds=[-float('inf'), float('inf')],
                                      name=kwargs.pop("name", "custom_gaussian"))

    def _pdf(self, x, *args):
        """!
        @brief Gamma PDF
        """
        m, s, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return (1.0 / np.sqrt(2.0 * np.pi * s ** 2.0)) * \
                np.exp(-(x - m) ** 2 / (2 * s ** 2.0))


    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        if params[1] <= 0.0:
            # sigam > 0
            return False
        return True
