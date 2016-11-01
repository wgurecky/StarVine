##
# \brief Uni-variate data container
#
from __future__ import absolute_import
import uvar.uvmodel_factory as uvf


class Uvd(object):
    """!
    @brief Container for univariate data.
    """
    def __init__(self, *args, **kwargs):
        self.fieldData = (args[0] if args else None)
        self.dataWeights = (args[1] if len(args) > 1 else None)
        self.dataName = kwargs.pop("dataName", None)
        self.boundingPlanes = kwargs.pop("boundingPlanes", None)
        self.uvmodel = uvf.Uvm("gauss")  # default gaussian univariate model

    def setUVM(self, uvModelName):
        """!
        @brief Select target univariate model.
        @param uvModelName <string>
               currently avalible models:  "gauss", "gamma", "beta"
        """
        self.uvmodel = uvf.Uvm(uvModelName)

    def setName(self, dataName):
        self.dataName = dataName

    def setWeights(self, dataWeights):
        self.dataWeights = dataWeights

    def setAxialBounds(self, boundingPlanes):
        self.boundingPlanes = boundingPlanes

    def setData(self, fieldData, dataWeights, boundingPlanes=None):
        """!
        @brief Set the univariate field data and weights
        @param fieldData    Scalar surface or volumetric field data
        @param dataWeights  weights of the data - typically areas or volumes
        @param boundingPlanes (optional) <tuple> of upper and lower axial bounds
        """
        self.fieldData = fieldData
        self.dataWeights = dataWeights
        self.boundingPlanes = boundingPlanes

    def fitMLE(self, params0=None):
        """!
        @brief fit model to data via maximum likelyhood.
        @param params0 Initial guess for model parameters.
               If None: use default initial guess
        """
        self.uvmodel.fitMLE(self.fieldData, params0, self.dataWeights)

    def fitGMM(self):
        pass

    def fitMCMC(self):
        pass
