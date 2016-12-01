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
                                          "id": i})
        self._setEdgeTriplets()

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
            for u, v in self.tree.edges_iter():
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
                # Set one fold triplet of each edge
                self.tree[u][v]['one-fold'] = \
                    (n_sides[0], n_sides[1], n_anchors[0])

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

    def _sampleEdge(self, n0, n1, old_n0, old_n1, size, vine):
        """!
        @brief Sample from edge in the tree.

            Current Tree:
            (n0)                                   (n1)
            Old_n0 ---------- Old_edge_0 --------- old_n1 ------------- Old_edge_1 -------------- old_n2
                                 :                                          :
            Next Tree:
                               ( old_n0 ) -------- edge_0 --------- ( old_n1 )

        To go "up" the vine requires evaluation of hinv-dist
        To traverse down the vine, evaluate the conditional h-dist function.

        @param n0  Node_0 in current tree (current edge connected)
        @param n1  Node_1 in current tree (current edge connected)
        @param old_n0  Node_0 from upperTree parent edge
        @param old_n0  Node_1 from upperTree
        @param size <b>int</b>  sample size
        """
        if type(n0) is int or type(n0) is np.int64 or type(n0) is str:
            tree_num = 0
        else:
            tree_num = len(n0) - 1
        current_tree = vine[tree_num].tree
        next_tree = vine[tree_num + 1].tree
        edge_info = current_tree[n0][n1]

        # if both marginal samples exist on this edge,
        # nothing to do - return.
        if edge_info.has_key('sample') and \
                len(edge_info['sample']) == 2:
            return

        # if u_n0 and u_n1 both dont exist, or if only u_n0 exists
        if not edge_info.has_key('sample') or not edge_info['sample'].has_key(n1):
            if tree_num == 0:
                u_n1 = np.random.rand(size)
            # if we are not in the first tree, try to get u_n1
            # from the H-function.A
            else:
                if not self.upperTree:
                    raise RuntimeError("Upper tree requested but unavalible.")
                prev_n0, prev_n1, prev_n2 = edge_info["one-fold"]
                prev_tree = vine[tree_num - 1].tree
                if not (prev_n0, prev_n2) == n1:
                    prev_n0 = prev_n1
                prev_edge_info = prev_tree[prev_n0][prev_n2]
                if prev_edge_info.has_key('sample') and \
                        len(prev_edge_info['sample']) == 2:
                    u_prev_n0 = prev_edge_info['sample'][prev_n0]
                    u_prev_n2 = prev_edge_info['sample'][prev_n2]
                    # u_n1 = prev_edge_info["h-dist"](u_prev_n0, u_prev_n2)
                    u_n1 = prev_edge_info["h-dist"](u_prev_n2, u_prev_n0)
                else:
                    u_n1 = np.random.rand(size)
        else:
            u_n1 = edge_info['sample'][n1]

        next_tree_info = next_tree[old_n0][old_n1]
        try:
            # u_n0 = next_tree_info["hinv-dist"](next_tree_info['sample'][(n0, n1)], u_n1)
            u_n0 = next_tree_info["hinv-dist"](u_n1, next_tree_info['sample'][(n0, n1)])
        except:
            # u_n0 = next_tree_info["hinv-dist"](next_tree_info['sample'][(n1, n0)], u_n1)
            u_n0 = next_tree_info["hinv-dist"](u_n1, next_tree_info['sample'][(n1, n0)])
        edge_sample = {n0: u_n0, n1: u_n1}
        current_tree[n0][n1]['sample'] = edge_sample

        # If current tree is 0th tree: copy marginal sample to
        # neighbor edge
        if tree_num == 0:
            for one_node in current_tree.neighbors(n0):
                if not current_tree[n0][one_node].has_key('sample'):
                    edge_sample = {n0:u_n0}
                    current_tree[n0][one_node]['sample'] = edge_sample
                else:
                    if not current_tree[n0][one_node]['sample'].has_key(n0):
                        current_tree[n0][one_node]['sample'][n0] = u_n0
            for one_node in current_tree.neighbors(n1):
                import pdb; pdb.set_trace()
                if not current_tree[n1][one_node].has_key('sample'):
                    edge_sample = {n1:u_n1}
                    current_tree[n1][one_node]['sample'] = edge_sample
                else:
                    if not current_tree[n1][one_node]['sample'].has_key(n1):
                        current_tree[n1][one_node]['sample'][n1] = u_n1
            return

        prev_n0, prev_n1, prev_n2 = edge_info['one-fold']
        self._sampleEdge(prev_n0, prev_n2, n0, n1, size, vine)
        self._sampleEdge(prev_n1, prev_n2, n0, n1, size, vine)
        return
