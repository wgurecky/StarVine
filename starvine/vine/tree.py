##
# \brief Generic tree structure.  Each level in a
# vine can be described as a tree.
#
from pandas import DataFrame
from starvine.bvcopula import pc_base as pc
import networkx as nx
import numpy as np


class Vtree(object):
    def __init__(self, data, lvl, parentTree=None, **kwargs):
        """!
        @brief A generic tree within vine.
        @param data <b>DataFrame</b>  multivariate data set. Each data
                                      column will be assigned to a node.
        @param lvl <b>int</b>: tree level in the vine
        @param weights <b>DataFrame</b>: (optional) data weights
        @param labels <b>list</b> of <b>str</b> or <b>ints</b>: (optional) data labels
        """
        assert(type(data) is DataFrame)
        assert(len(data.shape) == 2)
        self.data = data
        self._upperTree = parentTree
        #
        self.nT = data.shape[1]
        self.level = lvl
        #
        self.tree = nx.Graph()
        self.buildNodes()  # Construct nodes
        self.setEdges()    # Build edges between nodes

    def addNode(self, dataLabel, data):
        """!
        @brief Add a node to the tree. Prevents adding duplicate nodes.
        @param dataLabel <b>int</b> or <b>str</b>
        @param data <b>np_1darray</b>
        """
        if dataLabel in self.tree.nodes():
            self.tree.node[dataLabel]["data"] = data
        else:
            self.tree.add_node(dataLabel, attr_dict={"data": data})

    def buildNodes(self):
        """!
        @brief Assign each data column to a networkx node.
        """
        for colName in self.data:
            self.tree.add_node(colName, attr_dict={"data": self.data[colName]})

    def setEdges(self, nodePairs=None):
        """!
        @brief Sets the node to node connections in the tree.
        @param nodePairs <b>list</b> List of len 3 <b>tuples</b>:

            [(dataLabel_1, dataLabel_2, kTau), ...]
        """
        nodePairs = self._optimNodePairs() if nodePairs is None else nodePairs
        for i, pair in enumerate(nodePairs):
            self.tree.add_edge(pair[0], pair[1], weight=(1. - pair[2]),
                               attr_dict={"pc":
                                          pc.PairCopula(self.data[pair[0]].values,
                                                        self.data[pair[1]].values),
                                          "id": i,
                                          },
                               )

    def _optimNodePairs(self):
        """!
        @brief Computes the minimum spanning tree (R-vine) or
        tree which maximizes dependence. (private method)

        Virtual function.
        @return <b>list</b> List of len 3 <b>tuples</b>:

            [(dataLabel_1, dataLabel_2, kTau), ...]
        """
        raise NotImplementedError

    def _importTree(self, existingTreeStruct):
        """!
        @brief Imports an existing tree.
        Checks tree for correctness.

        @param existingTreeStruct <b>list</b>: list of tuples.
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

    def _exportTree(self):
        """!
        @brief Export tree for later use or storage.
        """
        pass

    @property
    def lowerTree(self, lowerTree=None):
        """!
        @brief Gets lower tree
        @returns starvine.vine.tree.Vtree object
        """
        return self._lowerTree

    @lowerTree.setter
    def lowerTree(self, lTree):
        """!
        @brief Sets lower tree
        """
        if type(lTree) != type(self):
            raise ValueError("Tree setter method takes tree type only.")
        self._lowerTree = lTree

    @property
    def upperTree(self, upperTree=None):
        """!
        @brief Gets upper tree
        @returns starvine.vine.tree.Vtree object
        """
        return self._upperTree

    @upperTree.setter
    def upperTree(self, uTree):
        """!
        @brief Sets upper tree
        """
        if type(uTree) != type(self):
            raise ValueError("Tree setter method takes tree type only.")
        self._upperTree = uTree

    def _sampleEdge(self, n0, n1, old_n0, old_n1, size):
        """!
        @brief Sample from edge in the tree.

            Old_n0 ---------- Old_edge_0 --------- old_n1 ------------- Old_edge_1 -------------- old_n2
                                 :                                          :
                         (old_u1 | Old_u0) ------- edge_0 --------- (old_u1 | old_u2)
                                n0                                         n1

        To go "up" the vine requires evaluation of hinv-dist
        To traverse down the vine, evaluate the conditional h-dist function.

        @param n0  Node_0 in current tree (current edge connected)
        @param n1  Node_1 in current tree (current edge connected)
        @param old_n0  Node_0 from upperTree parent edge
        @param old_n0  Node_1 from upperTree
        @param size <b>int</b>  sample size
        """
        current_tree = self.tree
        next_tree = self._lowerTree
        edge_info = current_tree[n0][n1]

        # if both marginal samples exist on this edge,
        # nothing to do - return.
        if edge_info.has_key('sample'):
            if len(edge_info['sample']) == 2:
                return

        # if u_n0 and u_n1 both dont exist, or if only u_n0 exists
        if not edge_info.has_key('sample') or not edge_info['sample'].has_key(n1):
            if self.level == 0:
                u_n1 = np.random.rand(size)
            # if we are not in the first tree, try to get u_n1
            # from the H-function.A
            else:
                if not hasattr(self, '_upperTree'):
                    raise RuntimeError("Upper tree requested but unavalible.")
                pass
        else:
            u_n1 = edge_info['sample'][n1]
        pass
