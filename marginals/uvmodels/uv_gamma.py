##
# @brief Gamma model distribution
from uv_base import UVmodel
import numpy as np
from math import gamma


class UVGamma(UVmodel):
    def __init__(self, *args, **kwargs):
        self.nMomentConditions = kwargs.pop("nMomentConditions", 2)

    def _pdf(self, x, params):
        a, b = params[0], params[1]
        return (b ** a) * x ** (a - 1) * np.exp(-x * b) / gamma(a)
