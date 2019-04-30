##
# @brief Beta model distribution

from .uv_base import UVmodel
import numpy as np
from scipy.special import gamma


class UVBeta(UVmodel):
    """!
    @brief Custom 2 parameter beta distribution.

    Note:
        Included example.
    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Beta PDF is supported on (0, 1)
        super(UVBeta, self).__init__(paramsString="a, b",
                                     momtype=0,
                                     bounds=[0.0, 1.0],
                                     name=kwargs.pop("name", "custom_beta"))
        self.defaultParams = [1.5, 5.]

    def _pdf(self, x, *args):
        """!
        @brief Beta PDF
        """
        a, b, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        beta_factor = gamma(a) * gamma(b) / gamma(a + b)
        return x ** (a - 1.0) * (1.0 - x) ** (b - 1.0) / beta_factor

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        for p in params:
            # all params >0
            if p <= 1e-6:
                return False
        return True
