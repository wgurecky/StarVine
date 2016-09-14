Vines
======

C- and D- vine copula classes are described in this directory.

Vines are nested tree data structures.  The nodes are univariate marginal distributions
and the edges are bivariate copula.

To construct a vine one must:

1) Determine the tree structure
2) Fit copula (maximum likelyhood) at all edges in the vine
    2.1) Fit marginal distributions at all nodes


C-Vine
=======

In each tree level in a C-Vine (Canonical Vine), a central node is selected to which all
other nodes are attached via copula.

This dependence structure is favorable if a single primary variable drives changes in all
other variables in the system to some degree.  This primary variable could then be
selected as the "anchor" node.

D-Vine
======

In a D-Vine, every node is connected with exactly two other nodes via copula.

This dependence structure is well suited when no single primary variable of interest
can be identified.


Regular Vines
=============

Regular vines (R-vine) are the superset of all possible vine-copula structures.  C- and D-vines
are rather simple examples of valid regular vines.

Arbitrary regular vine inference is not implemented in StarVine.  See packages:

- pyvine : A python library for regular vine modeling
- CDvine : A R library for C,D,and R-vine modeling
