##
# \brief Generic tree structure.  Each level in a
# vine can be described as a tree.
#
from pandas import DataFrame
from itertools import chain
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
        self.trial_copula_dict = kwargs.get("trial_copula", {})
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
            self.tree.nodes[dataLabel]["data"].update(data)
        else:
            self.tree.add_node(dataLabel, data=data)

    def buildNodes(self):
        """!
        @brief Assign each data column to a networkx node.
        """
        for colName in self.data:
            self.tree.add_node(colName, data=self.data[colName])

    def setEdges(self, nodePairs=None):
        """!
        @brief Sets the node to node connections in the tree.
        @param nodePairs <b>list</b> List of len 3 <b>tuples</b>:

            [(dataLabel_1, dataLabel_2, kTau), ...]
        """
        nodePairs = self._optimNodePairs() if nodePairs is None else nodePairs
        for i, pair in enumerate(nodePairs):
            self.tree.add_edge(pair[0], pair[1], weight=pair[2],
                                          pc= \
                                          pc.PairCopula(self.tree.nodes[pair[0]]["data"],
                                                        self.tree.nodes[pair[1]]["data"],
                                                        id=(pair[0], pair[1]),
                                                        family=self.trial_copula_dict),
                                          id=(pair[0], pair[1]),
                                          edge_data={pair[0]: self.tree.nodes[pair[0]]["data"],
                                                     pair[1]: self.tree.nodes[pair[1]]["data"]},
                              )
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
                pc.PairCopula(self.tree.nodes[pair[0]]["data"].values,
                              self.tree.nodes[pair[1]]["data"].values)
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

            Sample edge marked XXX:
            ### Current Tree ###
            (prev_n0)          XXX             (prev_n2)                              (prev_n1)
                n0 ---------- edge_0 ------------- n1 ------------- edge_1 -------------- n0
                                 :                                    :
            ### Next Tree ###
                          (next_u_n0) -------- next_edge --------- (next_u_n1)

        To go "up" the vine requires evaluation of hinv-dist
        To traverse down the vine, evaluate the conditional h-dist function.

        @param n0  Wing node in current tree
        @param n1  central node in current tree
        @param old_n0  Node_0 from lowerTree edge
        @param old_n1  Node_1 from lowerTree
        @param size <b>int</b>  sample size
        """
        def unrollNodes(l):
            try:
                flt = list(chain.from_iterable(l))
            except:
                return l
            return unrollNodes(flt)

        if type(n0) is int or type(n0) is np.int64 or type(n0) is str:
            tree_num = 0
        else:
            # Determine tree level from number of nodes (#nodes are powers of 2)
            # tree_num = int(np.log(len(unrollNodes(n0))) / np.log(2)) - 1
            tree_num = len(np.unique(np.asarray(n0).flatten())) - 1
        current_tree = vine[tree_num].tree
        next_tree = vine[tree_num + 1].tree
        edge_info = current_tree[n0][n1]

        # if both marginal samples exist on this edge,
        # nothing to do.
        if 'sample' in edge_info and \
                len(edge_info['sample']) == 2:
            return

        # if u_n0 and u_n1 both dont exist, or if only u_n0 exists
        if 'sample' not in edge_info or n1 not in edge_info['sample']:
            if tree_num == 0:
                u_n1 = np.random.rand(size)
            # if we are not in the first tree:
            # try to get u_n1 from the H-function
            else:
                if not self.upperTree:
                    raise RuntimeError("Upper tree requested but unavalible.")
                prev_n0, prev_n1, prev_n2 = edge_info["one-fold"]
                prev_tree = vine[tree_num - 1].tree
                if not (prev_n0, prev_n2) == n1:
                    prev_n0 = prev_n1
                prev_edge_info = prev_tree[prev_n0][prev_n2]
                if 'sample' in prev_edge_info and \
                        len(prev_edge_info['sample']) == 2:
                    u_prev_n0 = prev_edge_info['sample'][prev_n0]
                    u_prev_n2 = prev_edge_info['sample'][prev_n2]
                    u_n1 = prev_edge_info["h-dist"](u_prev_n2, u_prev_n0)
                else:
                    u_n1 = np.random.rand(size)
        else:
            u_n1 = edge_info['sample'][n1]

        next_tree_info = next_tree[old_n0][old_n1]
        try:
            u_n0 = edge_info["hinv-dist"](u_n1, next_tree_info['sample'][(n0, n1)])
        except:
            # u_n0 = edge_info["hinv-dist"](u_n1, next_tree_info['sample'][(n1, n0)])
            raise RuntimeError("Edge with nodes: " + str((n0, n1)), " does not exist.")
        edge_sample = {n0: u_n0, n1: u_n1}
        current_tree[n0][n1]['sample'] = edge_sample

        # If current tree is 0th tree: copy marginal sample to
        # neighbor edge
        if tree_num == 0:
            for one_node in current_tree.neighbors(n0):
                if 'sample' not in current_tree[n0][one_node]:
                    edge_sample = {n0:u_n0}
                    current_tree[n0][one_node]['sample'] = edge_sample
                else:
                    if n0 not in current_tree[n0][one_node]['sample']:
                        current_tree[n0][one_node]['sample'][n0] = u_n0
            for one_node in current_tree.neighbors(n1):
                if 'sample' not in current_tree[n1][one_node]:
                    edge_sample = {n1:u_n1}
                    current_tree[n1][one_node]['sample'] = edge_sample
                else:
                    if n1 not in current_tree[n1][one_node]['sample']:
                        current_tree[n1][one_node]['sample'][n1] = u_n1
            return

        # Traverse up the vine one level
        prev_n0, prev_n1, prev_n2 = edge_info['one-fold']
        self._sampleEdge(prev_n0, prev_n2, n0, n1, size, vine)
        self._sampleEdge(prev_n1, prev_n2, n0, n1, size, vine)
        return
