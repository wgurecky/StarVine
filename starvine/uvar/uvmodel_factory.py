##
# \brief Univariate model factory class
from uvmodels import *


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
        elif uvType == "logitnorm":
            return uv_logit_norm.UVLogitNorm()
        elif uvType == "exp":
            return uv_exp.UVExp()
        elif uvType == "gamma":
            return uv_gamma.UVGamma()
        elif uvType == "beta":
            return uv_beta.UVBeta()
        elif uvType == "exppow":
            return uv_exppow.UVexppow()
        else:
            # default
            raise RuntimeError("Model %s is not available" % str(copulatype))

    def __init__(self, uvType):
        self.uvType = uvType
