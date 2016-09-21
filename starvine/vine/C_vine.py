##
# \brief C-Vine structure.
# At each level in the C-vine, the tree structure is
# selected by searching for the parent node which maximizes
# the sum of all edge-weights.  The edge weights are taken to
# be abs(empirical kendall's tau) correlation coefficients in this
# implementation.
#
from base_vine import BaseVine
import networkx as nx


class Cvine(BaseVine):
    """!
    The nodes of the top level Tree are
    """
    def __init__(self, data):
        self.data = data
        self.levels = data.shape[1]
        self.vine = nx.Graph()

    def constructVine(self):
        """!
        @brief Sequentially construct the vine structure.
        Construct the top-level tree first.
        """
        pass

    def vineLLH(self):
        """!
        @brief Compute the vine log likelyhood.  Used for
        simulatneous MLE estimation of PCC model parameters.
        """
        pass


class Ctree(object):
    """!
    @brief A C-tree is a tree with a single root node.
    Each level of a cononical vine is a C-tree.
    """
    def __init__(self, nT):
        """!
        @brief
        @param nT <int> number of variables in tree.
        """
        self.nT = nT
        self.tree = nx.Graph()

    def selectSpanningTree(self):
        """!
        @brief Selects the tree which maximizes the sum
        over all edge weights, with the constraint that the
        tree is a C-tree stucture.

        It is feasible to try all C-tree configuations
        since if we have nT variables in the top level tree
        the number of _unique_ C-trees is == nT.
        """
        pass
