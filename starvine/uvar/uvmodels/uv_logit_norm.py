##
# @brief Logit distirbution

from .uv_base import UVmodel
import numpy as np
from scipy.special import erf, logit


class UVLogitNorm(UVmodel):
    """!
    @brief Custom 2 parameter logit gaussian distribution.
    """
    def __init__(self, *args, **kwargs):
        super(UVLogitNorm, self).__init__(paramsString="m, s",
                                          momtype=0,
                                          bounds=[0.0, 1.0],
                                          name=kwargs.pop("name", "custom_logit"))
        self.defaultParams = [0.0, 0.32]

    def _pdf(self, x, *args):
        """!
        @brief Logit-Norm PDF
        """
        m, s, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        c1 = 1. / (s * np.sqrt(2 * np.pi))
        c2 = 1. / (x * (1 - x))
        ex = np.exp(-(logit(x) - m) ** 2 / (2. * s ** 2))
        return c1 * c2 * ex

    def _cdf(self, x, *args):
        """!
        @brief Logit-Norm CDF
        """
        m, s, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return 0.5 * (1 + erf((logit(x) - m) / np.sqrt(2. * s ** 2.)))

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        if params[1] <= 0:
            return False
        return True
