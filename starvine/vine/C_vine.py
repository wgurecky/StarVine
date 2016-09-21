##
# \brief C-Vine structure.
# At each level in the C-vine, the tree structure is
# selected by searching for the parent node which maximizes
# the sum of all edge-weights.  The edge weights are taken to
# be abs(empirical kendall's tau) correlation coefficients in this
# implementation.
#
from base_vine import BaseVine


class Cvine(BaseVine):
    def __init__(self):
        pass


class Ctree(object):
    """!
    @brief A C-tree is a tree with a single root node.
    Each level of a cononical vine is a C-tree.
    """
    def __init__(self, nT, depth):
        """!
        @brief
        @param nT <int> number of variables in top level tree.
        @param depth <int>  on [1, nT - 1]. Depth of tree in the vine.
        """
        self.nT = nT
        self.depth = depth

    def _selectSpanningTree(self):
        """!
        @brief Selects the tree which maximizes the sum
        over all edge weights, with the constraint that the
        tree is a C-tree stucture.

        It is feasible to try all C-tree configuations
        since if we have nT variables in the top level tree
        the number of _unique_ C-trees is == nT.
        """
        pass
