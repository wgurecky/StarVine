##
# \brief C-Vine structure.
# At each level in the C-vine, the tree structure is
# selected by searching for the parent node which maximizes
# the sum of all edge-weights.  The edge weights are taken to
# be abs(empirical kendall's tau) correlation coefficients in this
# implementation.
#
from base_vine import BaseVine
from pandas import DataFrame
import networkx as nx
from starvine.bvcopula import pc_base as pc
import numpy as np


class Cvine(BaseVine):
    """!
    The nodes of the top level Tree are
    """
    def __init__(self, data):
        self.data = data
        self.nLevels = int(data.shape[1] - 1)
        self.vine = nx.Graph()

    def constructVine(self):
        """!
        @brief Sequentially construct the vine structure.
        Construct the top-level tree first.
        """
        # In the top level tree, each node is univariate data set
        # Each edge stores a nested nx.Graph() & has weight
        # == to kendall's tau
        pass

    def vineLLH(self, plist, **kwargs):
        """!
        @brief Compute the vine log likelyhood.  Used for
        simulatneous MLE estimation of PCC model parameters.
        """
        pass

    def treeHfun(self, level=0):
        """!
        @brief Operates on a tree, T_(i).
        The dependent distribution is evaluated
        at each edge in the tree providing univariate distributions that
        populate the dataFrame in the tree level T_(i+1)
        """
        pass


class Ctree(object):
    """!
    @brief A C-tree is a tree with a single root node.
    Each level of a cononical vine is a C-tree.
    """
    def __init__(self, data, lvl=None, **kwargs):
        """!
        @brief A single tree withen the overall vine.
        @param data <DataFrame>  multivariate data set, each data
            column will be assigned to a node.
        @param lvl <int> tree level in the vine
        @param weights <DataFrame> (optional) data weights
        """
        assert(len(self.data.shape) == 2)
        if type(data) is DataFrame:
            self.data = data
        else:
            self.data = DataFrame(data)
            # default data column labels are integers
            dataLabels = kwargs.pop("labels", range(data.shape[1]))
            self.data.columns = dataLabels
        self.nT = data.shape[1]
        self.level = lvl
        #
        self.tree = nx.Graph()
        self._buildNodes()

    def setEdges(self, treeStructure=None):
        """!
        @brief Computes the optimal C-tree structure based on maximizing
        kendall's tau summed over all edges.
        Optionally accepts a user input for the C-tree structure.
        @param treeStructure <list of <tuples>> pairs of nodes which must
            adhear to C-tree structure
        """
        if treeStructure:
            nodePairs = self._setTreeStructure(treeStructure)
        else:
            nodePairs = self._selectSpanningTree()
        for pair in nodePairs:
            self.tree.add_edge(pair[0], pair[1], weight=pair[2],
                               attr_dict={"pc":
                                          pc.PairCopula(self.data[pair[0]].values,
                                                        self.data[pair[1]].values)
                                          },
                               )

    def _buildNodes(self):
        """!
        @brief Assign each data column to a node
        """
        for colName in self.data:
            self.tree.add_node(colName, attr_dict={"data": self.data[colName]})

    def _setTreeStructure(self, nodePairs):
        """!
        @brief Checks tree pairs for correctness
        """
        # Check that each pair contains the root node
        # compute edge weights
        treeStructure = []
        for pair in nodePairs:
            trialPair = \
                pc.PairCopula(self.tree.node[pair[0]]["data"].values,
                              self.tree.node[pair[1]]["data"].values)
            treeStructure.append((pair[0], pair[1], trialPair.empKTau()[0]))
        return treeStructure

    def _selectSpanningTree(self):
        """!
        @brief Selects the tree which maximizes the sum
        over all edge weights provided the constraint of
        a C-tree stucture.

        It is feasible to try all C-tree configuations
        since if we have nT variables in the top level tree
        the number of _unique_ C-trees is == nT.
        @return <nd_array> [nT-1, 2] size array with PCC pairs
        """
        nodeIDs = self.data.columns
        trialKtauSum = np.zeros(len(nodeIDs))
        trialPairings = []
        # Generate all possible node pairings
        for i, rootNodeID in enumerate(nodeIDs):
            trialPairings.append([])
            for nodeID in nodeIDs:
                # iterate though all child nodes
                if nodeID is not rootNodeID:
                    trialPair = pc.PairCopula(self.tree.node[rootNodeID]["data"].values,
                                              self.tree.node[nodeID]["data"].values)
                    trialKtau, trialP = trialPair.empKTau
                    trialKtauSum[i] += trialKtau
                    trialPairings[i].append((rootNodeID, nodeID, trialKtau))
                else:
                    # root dataset cannot be paired with itself
                    pass
        # Find max trialKtauSum
        bestPairingIndex = np.argmax(np.abs(trialKtauSum))
        return trialPairings[bestPairingIndex]

    def fitCopula(self):
        """!
        @brief Iterate through all edges in tree, fit copula models
        at each edge.
        """
        for u, v, data in self.tree.edges(data=True):
            # (copulaModel <Copula>, copulaParams <list>)
            # TODO: Freeze copula model and parameters after
            # copula fit
            self.tree.edge[u][v]['pc-model'] = \
                data["pc"].copulaTournament()

    def treeLLH(self):
        """!
        @brief Compute this tree's pair copula construction log likelyhood.
        For C-trees this is just the sum of copula-log-likeyhoods over all
        node-pairs.
        """
        pass

    def evalH(self):
        """!
        @brief Define nodes of the next level tree.  Use the dependence function
        ("H" function) to obtain marginal distributions at the next tree level.
        """
        for u, v, data in self.tree.edges(data=True):
            self.tree.edge[u][v]["h-dist"] = \
                data["pc-model"][0].h(self.tree.node[u], self.tree.node[v], 0, data["pc-model"][1])
