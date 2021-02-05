##
# \brief R-Vine structure.
# At each level in the R-vine, the tree structure is
# selected by searching for the maximum spanning tree
# The edge weights are taken to
# be abs(empirical kendall's tau) correlation coefficients in this
# implementation.
#
from starvine.vine.base_vine import BaseVine
from pandas import DataFrame
from scipy.optimize import minimize
import networkx as nx
from starvine.bvcopula import pc_base as pc
import numpy as np
from six import iteritems
from starvine.vine.tree import Vtree


class Rvine(BaseVine):
    """!
    @brief Regular vine (R-vine).  Provides methods to fit pair
    copula sequentially and simultaneously.
    Additional methods are provided to draw samples from a
    constructed R-vine.
    Base class starvine.vine.base_vine.BaseVine .
    """
    def __init__(self, data, dataWeights=None, **kwargs):
        super(Rvine, self).__init__(data, dataWeights, **kwargs)
        self.nLevels = int(data.shape[1] - 1)
        self.vine = []

    def constructVine(self):
        """!
        @brief Sequentially construct the vine structure.
        Construct the top-level tree first, then recursively
        build all tree levels.
        """
        # 0th tree build
        tree0 = Rtree(self.data, lvl=0, trial_copula=self.trial_copula_dict)
        tree0.seqCopulaFit()
        self.vine.append(tree0)
        # build all other trees
        self.buildDeepTrees()

    def buildDeepTrees(self, level=1):
        """!
        @brief Recursivley build each tree in the vine.
        Must keep track of edge---node linkages between trees.
        @param level <b>int</b> Current tree level.
        """
        treeT = Rtree(self.vine[level - 1].evalH(),
                      lvl=level,
                      parentTree=self.vine[level - 1],
                      trial_copula=self.trial_copula_dict)
        treeT.seqCopulaFit()
        if self.nLevels > 1:
            self.vine.append(treeT)
        if level < self.nLevels - 1:
            self.buildDeepTrees(level + 1)
        elif level == self.nLevels - 1:
            self.vine[level].evalH()


class Rtree(Vtree):
    """!
    @brief A C-tree is a tree with a single root node.
    Each level of a canonical vine is a C-tree.
    """
    def __init__(self, data, lvl=None, **kwargs):
        """!
        @brief A single tree within vine.
        @param data <b>DataFrame</b>  multivariate data set. Each data
                                      column will be assigned to a node.
        @param lvl <b>int</b>: tree level in the vine
        @param weights <b>DataFrame</b>: (optional) data weights
        @param labels <b>list</b> of <b>str</b> or <b>ints</b>: (optional) data labels
        """
        super(Rtree, self).__init__(data, lvl, **kwargs)

    def seqCopulaFit(self):
        """!
        @brief Iterate through all edges in tree, fit copula models
        at each edge.  This is a sequential fitting operation.

        See simultaneousCopulaFit() for a tree-wide simulltaneous
        parameter estimation.
        """
        for u, v, data in self.tree.edges(data=True):
            data["pc"].copulaTournament()
        self._initTreeParamMap()

    def treeNLLH(self, treeCopulaParams=None):
        """!
        @brief Compute this tree's negative log likelihood.
        For C-trees this is just the sum of copula-log-likelihoods over all
        node-pairs.
        @param treeCopulaParams <b>np_1darray</b> Copula parameter array.
        Contains parameters for all PCC in the tree.

        @param paramMap  Maps edges to parameter len and location in
        treeCopulaParams
        {[u, v]: (start, Params_len_0), [u, v]: (start, Params_len), ...}
        """
        if not treeCopulaParams:
            if not hasattr(self, "treeCopulaParams"):
                raise RuntimeError("Sequential copula model fitting must be performed first.")
            treeCopulaParams = self.treeCopulaParams
        nLL = 0
        for u, v, data in self.tree.edges(data=True):
            nLL += \
                self.tree.adj[u][v]["pc"].copulaModel.\
                _nlogLike(self.tree.nodes[u]["data"].values,
                          self.tree.nodes[v]["data"].values,
                          0,
                          *treeCopulaParams[data["paramMap"][0]: data["paramMap"][1]])
        return nLL

    def evalH(self):
        """!
        @brief Define nodes of the T+1 level tree.  Use the conditional distribution
        (h()) to obtain marginal distributions at the next tree level.
        """
        for u, v, data in self.tree.edges(data=True):
            self.tree.adj[u][v]["h-dist"] = data["pc"].copulaModel.h
            self.tree.adj[u][v]["hinv-dist"] = data["pc"].copulaModel.hinv
        return self._evalH()

    # ---------------------------- PRIVATE METHODS ------------------------------ #
    def _optimNodePairs(self):
        """!
        @brief Selects the node-pairings which maximizes the sum
        over all edge weights using a maximum spanning tree algorithm.

        @return <b>nd_array</b>: (nT-1, 3) shape array with PCC pairs
                Each row is a len 3 tuple: (rootNodeID, nodeID, kTau)
        """
        fully_connected_tree = nx.generators.classic.complete_graph(
                self.tree.nodes())
        if self.upperTree is not None:
            print(self.upperTree.tree.edges)
        print(fully_connected_tree.edges)
        for e in fully_connected_tree.edges():
            trialPair = pc.PairCopula(self.tree.nodes[e[0]]["data"].values,
                                      self.tree.nodes[e[1]]["data"].values)
            trialKtau, trialP = trialPair.empKTau()
            # assign emperical calculated ktau as edge weights
            fully_connected_tree[e[0]][e[1]]['weight'] = abs(trialKtau)
            fully_connected_tree[e[0]][e[1]]['ktau'] = trialKtau

            if self.upperTree is not None:
                if set(e[0]).intersection(set(e[1])):
                    pass
                else:
                    fully_connected_tree[e[0]][e[1]]['weight'] = -1e10
        # compute max spanning tree
        mst = nx.algorithms.tree.maximum_spanning_tree(
                fully_connected_tree, weight='weight')
        # mst_edge_list = sorted(list(mst.edges()))
        mst_edge_list = list(mst.edges())
        trialPairings = list([tuple([e_mst[0], e_mst[1], fully_connected_tree.edges[e_mst]['ktau']]) for e_mst in mst_edge_list])
        return trialPairings

    def _evalH(self):
        """!
        @brief Computes \f$ F(x|v, \theta) \f$ data set at each node
        for use in the next level tree.
        @return <b>DataFame</b> : conditional distribution at tree edges.
        """
        # TODO: Establish linkage between tree levels
        condData = DataFrame()
        for u, v, data in self.tree.edges(data=True):
            if self.upperTree is not None:
                pass
            condData[(u, v)] = data["h-dist"](data["pc"].UU,
                                              data["pc"].VV)
        return condData

    def _setEdgeTriplets(self):
        """!
        @brief Applies to all non-zero level trees in the vine.
        Every edge in the tree has 3 spanning nodes in the
        tree above.  These nodes comprise a 'one fold triplet'.
        Sets the one fold triplet for each edge in current tree.
        """
        if self.upperTree is None:
            return
        else:
            for u, v in self.tree.edges():
                # every edge has data nodes (u, v)
                # u and v  are tuples
                # each node came from an edge above
                # Obtain unique nodes, non-unique upper node
                # is the "anchor" node of the one fold triplet
                upper_nodes = list((u[0], u[1], v[0], v[1]))
                n_sides, n_anchors = [], []
                for node in upper_nodes:
                    # check for common node
                    if upper_nodes.count(node) == 2:
                        n_anchors.append(node)
                    else:
                        n_sides.append(node)
                assert(n_anchors[0] == n_anchors[1])
                anchor = n_anchors[0]
                # Set one fold triplet of each edge
                #print("===One-fold-builder===")
                #print("===",n_sides[0], "--|--", n_anchors[0], "--|--",  n_sides[1])
                #print("===","|", u, "|", v)
                if u[1] != anchor and v[1] != anchor:
                    self.tree[u][v]['one-fold'] = \
                        (u[1], v[1], anchor)
                elif u[0] != anchor and v[1] != anchor:
                    self.tree[u][v]['one-fold'] = \
                        (u[0], v[1], anchor)
                elif u[1] != anchor and v[0] != anchor:
                    self.tree[u][v]['one-fold'] = \
                        (u[1], v[0], anchor)
                elif u[0] != anchor and v[0] != anchor:
                    self.tree[u][v]['one-fold'] = \
                        (u[0], v[0], anchor)
                else:
                    raise RuntimeError
                # print("===", self.tree[u][v]['one-fold'])

    def _getEdgeCopulaParams(self, u, v):
        """!
        @brief Get copula paramters of particular edge in tree.
        @returns <b>np_1darray</b> Copula parameters of edge
        """
        cp = self.tree.adj[u][v]["pc"].copulaParams
        if cp is not None:
            return cp
        else:
            raise RuntimeError("ERROR: Must execute sequential fit first")

    def _initTreeParamMap(self):
        """!
        @brief Pack all copula paramters in the tree into a 1d numpy array for
        simulatneous MLE optimization.  Sets the tree copula paramters.
        """
        currentMarker = 0
        self.treeCopulaParams = []
        edgeList = self.tree.edges(data=True)
        for u, v, data in edgeList:
            edgeParams = self._getEdgeCopulaParams(u, v)
            self.treeCopulaParams.append(edgeParams[1])
            nEdgeParams = len(edgeParams[1])
            self.tree.adj[u][v]["paramMap"] = \
                [currentMarker, currentMarker + nEdgeParams]
            currentMarker += nEdgeParams
        self.treeCopulaParams = [item for sublist in self.treeCopulaParams
                                 for item in sublist]
        self.treeCopulaParams = np.array(self.treeCopulaParams)
        return self.treeCopulaParams
