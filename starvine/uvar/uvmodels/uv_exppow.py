##
# @brief Exponentialpower model distribution

from .uv_base import UVmodel
import numpy as np
from scipy.special import gamma


class UVexppow(UVmodel):
    """!
    @brief Custom 2 parameter exponential power distribution.
    see:
        www.math.wm.edu/~leemis/chart/UDR/PDFs/Exponentialpower.pdf
    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Exponentialpower PDF is supported on (0, 1)
        super(UVexppow, self).__init__(paramsString="a, b",
                                       momtype=0,
                                       bounds=[0.0, np.inf],
                                       name=kwargs.pop("name", "custom_exppow"))
        self.defaultParams = [2., 1.]

    def _pdf(self, x, *args):
        """!
        @brief Exponentialpower PDF
        """
        a, b, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        exp1 = np.exp(1. - np.exp(a * x ** b))
        exp2 = np.exp(a * x ** b)
        powt = a * b * x ** (b - 1.)
        return exp1 * exp2 * powt

    def _cdf(self, x, *args):
        """!
        @brief Exponentialpower CDF
        """
        a, b, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return 1. - np.exp(1. - np.exp(a * x ** b))

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        for p in params:
            # all params >0
            if p <= 1e-6:
                return False
        return True
