##
# \brief Univariate model factory class
from uvmodels import *
import sys


class Uvm(object):
    """!
    @brief Sinple univariate model factory.
    """
    def __new__(cls, uvType):
        """!
        @brief Instantiates a univariate model object.
        @param uvType <b>str</b> Univariate model name.
        """
        if uvType == "gauss":
            return uv_gauss.UVGauss()
        elif uvType == "lognorm":
            return uv_lognorm.UVLogNorm()
        elif uvType == "gamma":
            return uv_gamma.UVGamma()
        elif uvType == "beta":
            return uv_beta.UVBeta()
        else:
            # default
            sys.exit("Invalid univariate data model name: " + str(uvType))

    def __init__(self, uvType):
        self.uvType = uvType
