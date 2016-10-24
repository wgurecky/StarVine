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
    @brief Cononical vine (C-vine).  Provides methods to fit pair
    copula constructions sequentially and simultaneously.
    Additional methods are provided to draw samples from a
    constructed C-vine.

    Example 3 variable C-vine structure
    +++++++++++++++++++++++++++++++++++

    Tree level 1
    -------
    ```
    X1 ---C_13--- X3
    |
    C_12
    |
    X2
    ```

    Tree level 2
    ------
    ```
    F(X2|X1) ---C_23|1--- F(X3|X1)
    ```

    The nodes of the top level tree are the rank transformed,
    uniformly distributed marginals (defined on [0, 1]).

    The formation of the lower level trees
    involve computing conditional distributions of the form:

    \f$ F(x|v) \f$

    For the bivariate case, Joe (1996) showed that:

    \f$ F(x|v) = \frac{\partial C_{x,v}(F(x), F(v))}{\partial F_v(v)} \f$

    Which simplifies further if x and v are uniform:

    \f$ h(x, v, \theta) = F(x|v) = \frac{\partial C_{xv}(x,v,\theta}{\partial v} \f$

    Where we have defined the convinience conditional distribution
    \f$ h(\cdot) \f$

    The nodes of the lower level trees are formed by using \f$ h(x,v,\theta) \f$
    to compute marginal distirbution of the RV \f$ x \f$ given the parent
    copula's parameters \f$ \theta \f$ and the root node's uniformly distributed
    \f$ v \f$.

    The n-dimensional density of a C-vine copula is given by:

    \f$ \prod_{k=1}^n f(x_k) \prod_{j=1}^{n-1} \prod_{i=1}^{n-j} c_{j,j+i|1,...j-1}(F(x_j|x_1...,x_{j-1}), F(x_{j+1}|x_1...,x_{j-1}))\f$

    Where the outer product index represents the tree level, and the inner product indicies represent
    the pair copula constructions withen the given tree.
    """
    def __init__(self, data):
        self.data = data
        self.nLevels = int(data.shape[1] - 1)
        self.vine = []

    def constructVine(self):
        """!
        @brief Sequentially construct the vine structure.
        Construct the top-level tree first, then recursively
        build all tree levels.
        """
        tree0 = Ctree(self.data, lvl=0)
        tree0.seqCopulaFit()
        self.vine.append(tree0)
        self.buildDeepTrees()

    def buildDeepTrees(self, level=1):
        """!
        @brief Recursivley build each tree in the vine.
        Must keep track of edge---node linkages between trees.
        """
        treeT = Ctree(self.tree[level - 1].evalH(), lvl=level)
        treeT.seqCopulaFit()
        self.vin.append(treeT)
        if level <= self.nLevels:
            self._computeTree(level + 1)

    def vineLLH(self, plist, **kwargs):
        """!
        @brief Compute the vine log likelyhood.  Used for
        simulatneous MLE estimation of PCC model parameters.
        """
        pass

    def treeHfun(self, level=0):
        """!
        @brief Operates on a tree, T_(i).
        The conditional distribution is evaluated
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
        @brief A single tree within vine.
        @param data <DataFrame>  multivariate data set. Each data
                                 column will be assigned to a node.
        @param lvl <int> tree level in the vine
        @param weights <DataFrame> (optional) data weights
        @param labels <list of str or ints> (optional) data labels
        """
        assert(type(data) is DataFrame)
        assert(len(self.data.shape) == 2)
        self.nT = data.shape[1]
        self.level = lvl
        #
        self.tree = nx.Graph()
        self.buildNodes()  # Construct nodes
        self.setEdges()    # Build edges between nodes

    def setEdges(self, existingTree=None):
        """!
        @brief Computes the optimal C-tree structure based on maximizing
        kendall's tau summed over all edges.
        @param existingTree (optional) A pre-computed C-tree
        """
        nodePairs = self._selectCTree()
        for pair in nodePairs:
            self.tree.add_edge(pair[0], pair[1], weight=pair[2],
                               attr_dict={"pc":
                                          pc.PairCopula(self.data[pair[0]].values,
                                                        self.data[pair[1]].values)
                                          },
                               )

    def buildNodes(self):
        """!
        @brief Assign each data column to a node
        """
        for colName in self.data:
            self.tree.add_node(colName, attr_dict={"data": self.data[colName]})

    def seqCopulaFit(self):
        """!
        @brief Iterate through all edges in tree, fit copula models
        at each edge.  This is a sequential fitting operation.

        See simultaneousCopulaFit() for a tree-wide simulltaneous
        parameter estimation.
        """
        for u, v, data in self.tree.edges(data=True):
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
        @brief Define nodes of the T+1 level tree.  Use the conditional distribution
        (h()) to obtain marginal distributions at the next tree level.
        """
        for u, v, data in self.tree.edges(data=True):
            self.tree.edge[u][v]["h-dist"] = data["pc"].copulaModel.h
        return self._evalH()

    # ---------------------------- PRIVATE METHODS ------------------------------ #
    def _importTree(self, existingTreeStruct):
        """!
        @brief Imports an existing tree.
        Checks tree pairs for correctness.
        """
        # Check that each pair contains the root node
        # compute edge weights
        treeStructure = []
        for pair in existingTreeStruct:
            trialPair = \
                pc.PairCopula(self.tree.node[pair[0]]["data"].values,
                              self.tree.node[pair[1]]["data"].values)
            treeStructure.append((pair[0], pair[1], trialPair.empKTau()[0]))
        return treeStructure

    def _selectCTree(self):
        """!
        @brief Selects the tree which maximizes the sum
        over all edge weights provided the constraint of
        a C-tree stucture.

        It is feasible to try all C-tree configuations
        since if we have nT variables in the top level tree
        the number of _unique_ C-trees is == nT.
        @return <nd_array> (nT-1, 3) shape array with PCC pairs
                Each row is a len 3 tuple: (rootNodeID, nodeID, kTau)
        """
        nodeIDs = self.data.columns
        trialKtauSum = np.zeros(len(nodeIDs))
        trialPairings = []
        # Generate all possible node pairings
        for i, rootNodeID in enumerate(nodeIDs):
            trialPairings.append([])
            for nodeID in nodeIDs:
                # iterate though all child nodes,
                # root dataset cannot be paired with itself
                if nodeID is not rootNodeID:
                    trialPair = pc.PairCopula(self.tree.node[rootNodeID]["data"].values,
                                              self.tree.node[nodeID]["data"].values)
                    trialKtau, trialP = trialPair.empKTau
                    trialKtauSum[i] += abs(trialKtau)
                    trialPairings[i].append((rootNodeID, nodeID, trialKtau))
        bestPairingIndex = np.argmax(np.abs(trialKtauSum))
        return trialPairings[bestPairingIndex]

    def _evalH(self):
        """!
        @brief Converts the univariate \f$ F(x|v) \f$ data sets at each node
        into a pandas data frame for use in the next level tree.
        @return <DataFrame> conditional distributions at tree edges.
        """
        condData = DataFrame()
        # linkage maps edges of T_(i) to nodes of tree T_(i+1)
        linkage = {}  # tracks T_(i)edge ---- T_(i+1)node linkage
        for u, v, data in self.tree.edges(data=True):
            # eval h() of pair-copula model at current edge
            condData[(u, v)] = data["h-dist"](self.tree.node[u]["data"],
                                              self.tree.node[v]["data"])
        return condData
