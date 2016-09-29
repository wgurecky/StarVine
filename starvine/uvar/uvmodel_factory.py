##
# \brief Concrete univariate model factory class
from uvmodels import *
import sys


class Uvm(object):
    def __new__(cls, uvtype):
        if uvtype is "gauss":
            return uv_gauss.UVGauss()
        elif uvtype is "gamma":
            return uv_gamma.UVGamma()
        elif uvtype is "beta":
            return uv_beta.UVBeta()
        else:
            # default
            sys.exit("Invalid univariate data model name.")

    def __init__(self, uvtype):
        self.uvtype = uvtype
