##
# \brief Generic tree structure.  Each level in a
# vine can be described as a tree.
#
from pandas import DataFrame
from starvine.bvcopula import pc_base as pc
import networkx as nx


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
        self.parentTree = parentTree
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
