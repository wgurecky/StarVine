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
from scipy.optimize import minimize
import networkx as nx
from starvine.bvcopula import pc_base as pc
import numpy as np
from tree import Vtree


class Cvine(BaseVine):
    """!
    @brief Cononical vine (C-vine).  Provides methods to fit pair
    copula constructions sequentially and simultaneously.
    Additional methods are provided to draw samples from a
    constructed C-vine.
    Base class starvine.vine.base_vine.BaseVine .

    Example 3 variable C-vine structure:

    ### Tree level 1 ###

        X1 ---C_13--- X3
        |
        C_12
        |
        X2

    ### Tree level 2 ###

        F(X2|X1) ---C_23|1--- F(X3|X1)

    The nodes of the top level tree are the rank transformed,
    uniformly distributed marginals (defined on [0, 1]).

    The formation of the lower level trees
    involve computing conditional distributions of the form:

    \f[ F(x|v) \f]

    For the bivariate case, Joe (1996) showed that:

    \f[ F(x|v) = \frac{\partial C_{x,v}(F(x), F(v))}{\partial F_v(v)} \f]

    Which simplifies further if x and v are uniform:

    \f[ h(x| v, \theta) = F(x|v, \theta) = \frac{\partial C_{xv}(x,v,\theta)}{\partial v} \f]

    Where we have defined the convinience conditional distribution
    \f$ h(\cdot) \f$

    The nodes of the lower level trees are formed by using \f$ h(x| v,\theta) \f$
    to compute marginal distirbution of the RV \f$ x \f$ given the parent
    copula's parameters \f$ \theta \f$ and the root node's uniformly distributed
    \f$ v \f$.

    The n-dimensional density of a C-vine copula is given by:

    \f[ \prod_{k=1}^n f(x_k) \prod_{j=1}^{n-1}
    \prod_{i=1}^{n-j} c_{j,j+i|1,...j-1}
    (F(x_j|x_1...,x_{j-1}), F(x_{j+1}|x_1...,x_{j-1}))\f]

    Where the outer product index represents the tree level,
    and the inner product indicies represent
    the pair copula constructions (PCC) withen the given tree.
    """
    def __init__(self, data, dataWeights=None):
        self.data = data
        self.weights = dataWeights
        self.nLevels = int(data.shape[1] - 1)
        self.vine = []

    def constructVine(self):
        """!
        @brief Sequentially construct the vine structure.
        Construct the top-level tree first, then recursively
        build all tree levels.
        """
        # 0th tree build
        tree0 = Ctree(self.data, lvl=0)
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
        treeT = Ctree(self.vine[level - 1].evalH(),
                      lvl=level,
                      parentTree=self.vine[level - 1])
        treeT.seqCopulaFit()
        if self.nLevels > 1:
            self.vine.append(treeT)
        if level < self.nLevels - 1:
            self.buildDeepTrees(level + 1)
        elif level == self.nLevels - 1:
            self.vine[level].evalH()


class Ctree(Vtree):
    """!
    @brief A C-tree is a tree with a single root node.
    Each level of a cononical vine is a C-tree.
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
        super(Ctree, self).__init__(data, lvl, **kwargs)

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
        @brief Compute this tree's negative log likelyhood.
        For C-trees this is just the sum of copula-log-likeyhoods over all
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
                _nlogLike(self.tree.node[u]["data"].values,
                          self.tree.node[v]["data"].values,
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
        over all edge weights provided the constraint of
        a C-tree stucture.

        It is feasible to try all C-tree configuations
        since if we have nT variables in the top level tree
        the number of _unique_ C-trees is == nT.

        @return <b>nd_array</b>: (nT-1, 3) shape array with PCC pairs
                Each row is a len 3 tuple: (rootNodeID, nodeID, kTau)
        """
        trialKtauSum = np.zeros(len(self.tree.nodes()))
        trialPairings = []
        # Generate all possible node pairings
        for i, rootNodeID in enumerate(self.tree.nodes()):
            trialPairings.append([])
            for nodeID in self.tree.nodes():
                # iterate though all child nodes,
                # root dataset cannot be paired with itself
                if nodeID != rootNodeID:
                    ## VV is OK UU is wrong!
                    trialPair = pc.PairCopula(self.tree.node[nodeID]["data"].values,
                                              self.tree.node[rootNodeID]["data"].values)
                    # trialPair = pc.PairCopula(self.tree.node[rootNodeID]["data"].values,
                    #                           self.tree.node[nodeID]["data"].values)
                    trialKtau, trialP = trialPair.empKTau()
                    trialKtauSum[i] += abs(trialKtau)
                    # trialPairings[i].append((rootNodeID, nodeID, trialKtau))
                    trialPairings[i].append((nodeID, rootNodeID, trialKtau))
        bestPairingIndex = np.argmax(np.abs(trialKtauSum))
        self.rootNodeID = trialPairings[bestPairingIndex][0][1]
        return trialPairings[bestPairingIndex]

    def _evalH(self):
        """!
        @brief Computes \f$ F(x|v, \theta) \f$ data set at each node
        for use in the next level tree.
        @return <b>DataFame</b> : conditional distribution at tree edges.
        """
        # TODO: Establish linkage between tree levels
        condData = DataFrame()
        for u, v, data in self.tree.edges(data=True):
            # eval h() of pair-copula model at current edge
            # use rank transformed data as input to conditional dist
            # identify rootID
            rootID = self.rootNodeID
            if u is not rootID:
                nonRootID = u
                nonRootData = data["pc"].UU
                rootData = data["pc"].VV
            else:
                nonRootID = v
                nonRootData = data["pc"].VV
                rootData = data["pc"].UU
            condData[(nonRootID, rootID)] = data["h-dist"](data["pc"].VV,
                                                           data["pc"].UU)
            # condData[(nonRootID, rootID)] = data["h-dist"](nonRootData,
            #                                                rootData)
        return condData

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
