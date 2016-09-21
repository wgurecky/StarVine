starvine
========

The starvine package is split into four subpackages:

- vine : C- and D- vine classes
    + Depends : (bvcopula, mvar)
- bvcopula : Bivariate copula
    + Depends : (uvar)
- mvar : Multi-variate data (N-dim data container)
- uvar : Uni-variate data (marginal distribution fitting)


A user may choose to only utilize the univariate fitting functions
of starvine, for example:

    from starvine import uvar as uv
