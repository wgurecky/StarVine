##
# \brief Pair copula construction.
# Stores copula model, two marginal disitribution models,
# aware of location in Vine,
# aware of node in next tree.
#
from starvine.bvcopula.pc_base import PairCopula
from starvine.uvar.uvd import Uvd


class PairCopulaConstruction(PairCopula):
    """!
    @brief Pair copula constructions are comprised of
    two "nodes" and one "edge" connecting them.
    The nodes are marginal distributions and the edge is a
    bivariate copula.

    @note The base implementation is provided by the
    @ref starvine.bvcopula.pc_base.PairCopula class.  This class extends the base
    class to provided awareness of the Vine's structure.
    """
    def __init__(self, *args, **kwargs):
        self.treeLevel = kwargs.pop("treeLevel", 0)
        super(self, PairCopulaConstruction).__init__(*args, **kwargs)

    @property
    def marginals(self):
        """!
        @brief Marginal distributions access.
        Marginal distribution models are stored in a tuple: (uModel, vModel)
        Ex maginal model access:

            >>> self.marginals[0].fitMLE()
            >>> self.marginals[0].parameters
        """
        # print("Getting marinals")
        return self._marginals

    @property.setter
    def setMarginals(self, uModelName, vModelName):
        """!
        @brief Set the marginal distribution models.
        @param uModel <b>str</b>  univariate model name of first marginal dist
        @param vModel <b>str</b>  univariate model name for second marginal dist
        """
        uModel = Uvd(fData=self.x, dWeights=self.weights, uvModelName=uModelName)
        vModel = Uvd(fData=self.x, dWeights=self.weights, uvModelName=vModelName)
        self._marginals = (uModel, vModel)

    def traverseDown(self):
        pass

    @property
    def lowerNode(self):
        """!
        @returns  Node in tree below this edge.
        """
        return 0

    @lowerNode.setter
    def setLowerNode(self):
        pass

    @property
    def neighborNodes(self):
        pass

    @neighborNodes.setter
    def setNeighborNodes(self, u, v):
        pass
