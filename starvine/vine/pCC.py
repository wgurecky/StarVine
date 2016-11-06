#!/usr/bin/python2
##
# \brief Pair copula construction.
# Stores copula model, two marginal disitribution models,
# aware of location in Vine,
# aware of node in next tree.
from starvine.bvcopula import pc_base as pc


def PairCopulaConstruction(pc.PairCopula):
    """!
    @brief Pair copula constructions are comprised of
    two "nodes" and one "edge" connecting them.
    The nodes are marginal distributions and the edge is a
    bivariate copula.
    @note:  The base implementation is provided by the
    PairCopula class.  This class extends the base
    class to provided awareness of the Vine's structure.
    """
    def __init__(self, *args, **kargs):
        self.treeLevel = kwargs.pop("treeLevel", 0)
        super(PairCopulaConstruction, self).__init__(*args, **kwargs)

    def genConditionalDist(self):
        pass

    def traverseDown(self):
        pass

    @property
    def lowerNode(self):
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
