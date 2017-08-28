[![Build Status](https://travis-ci.org/wgurecky/StarVine.svg?branch=master)](https://travis-ci.org/wgurecky/StarVine)

About
========

StarVine provides tools to construct canonical and regular-vines
(C-vines, and R-vines).

StarVine can also be used as a standalone copula fitting tool for bivariate modeling.

Install
========

Install the package:

    python setup.py install

For a developer install:

    python setup.py develop --user

Requires:

- A fortran compiler
- numpy : basic linear algebra operations
- scipy : additional numerical operations
- networkx : vine graph structure
- h5py : file I/O
- emcee : MCMC
- seaborn : plotting
- matplotlib : basic plotting, required by seaborn
- pandas

Optional:

- statsmodels: Generalized method of moments fitting
- mpi4py : parallel MCMC

Docs
=====

Documentation can be found online at:
[wgurecky.github.io/starvine](https://wgurecky.github.io/StarVine).

The docs can be manually built using Doxygen:

    doxygen Doxyfile.in

Similar Projects
----------------

- [VineCopulaCPP](https://github.com/MalteKurz/VineCopulaCPP)
- [VineCopula](https://github.com/tnagler/VineCopula)
- [CDvine](https://github.com/cran/CDVine)
- [pyvine](https://pypi.python.org/pypi/pyvine/0.5.0)


License
========

StarVine is distributed under the BSD 3 clause license.

A copy of the license should have been distributed with the StarVine source code.
If not, see [opensource.org](https://opensource.org/licenses/BSD-3-Clause)
