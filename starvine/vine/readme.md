Vines
======

C- and D- vine copula classes are described in this directory.

Vines are nested tree data structures.  The nodes are univariate marginal distributions
and the edges are bivariate copula.

To construct a vine sequentially starting from the top tree:

    For each level in the vine:
        i) Determine the tree structure.  In the case of a regular vine, this is the
           maximum spanning tree with edge weights equal to kendall's tau.
        ii) Fit copula (maximum likelyhood) at all edges in the vine


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
can be identified.  It is also suitable for temporally ordered data.


Regular Vines
=============

Regular vines (R-vine) are the superset of all possible vine-copula structures.  C- and D-vines
are rather simple examples of valid regular vines.

Arbitrary regular vine inference is not implemented in StarVine.  See packages:

- pyvine : A python library for regular vine modeling
- CDvine : A R library for C,D,and R-vine modeling
