About
========

[StarVine](starvine/readme.md) provides tools to construct canonical and regular-vines
(C-vines, and R-vines).

StarVine can also be used as a standalone copula fitting tool for bivariate modeling.

Install
========

Prep package:

    python setup.py build

Execute unit tests:

    python -m unittest discover

If everything checks out install the package:

    python setup.py install --user

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

Build the docs:

    doxygen Doxyfile.in

Requires:

- Doxygen

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
