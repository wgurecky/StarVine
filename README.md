About
========

StarVine provides tools to construct C- and D-vines.

StarVine can also be used as a standalone copula fitting tool.

Similar Projects
----------------

- [VineCopulaCPP](https://github.com/MalteKurz/VineCopulaCPP)
- [VineCopula](https://github.com/tnagler/VineCopula)
- [CDvine](https://github.com/cran/CDVine)
- [pyvine](https://pypi.python.org/pypi/pyvine/0.5.0)

Install
========

Prep package:

    python setup.py build

Execute unit tests:

    python -m unittest discover

To install a development version:

    python setup.py develop --user

Requires:

- A fortran compiler
- numpy : basic linear algebra operations
- scipy : additional numerical operations
- networkx : vine graph structure
- pandas : data analysis
- h5py : file I/O
- emcee : MCMC
- seaborn : plotting
- matplotlib : basic plotting, required by seaborn

Optional:

- statsmodels: Generalized method of moments fitting (optional)
- mpi4py : parallel MCMC (optional)

Docs
=====

Documentation is provided in the `/doc` directory.

Build the docs from `starvine` directory:

    doxygen Doxyfile.in

Requires:

    - Doxygen

Example
========

Read in some sample data:

    import starvine as sv
    mvdata = sv.mvd()
    mvdata.read_h5('example/ex_data.h5', '/group1/dset1')

Visualize the example multivariate data:

    fig1 = mvdata.pairPlot()
    fig1.show()
    # if the above fails or no X11 avalible try:
    # fig1.savefig('ex_out.png')

Set list of trial copula (default is try all implemented copula - very time consuming):

    trial_copula = ['t', 'normal', 'frank', 'clayton']

Decompose the data into a C-vine and visualize the tree structure:

    cvine = sv.cvine(mvdata, trial_copula)
    fig2 = cvine.plotGraph()
    fig2.show()

Visually compare the MLE fitted C-vine copula to raw empirical data.

    cvine.simulate(n=1000)
    # plot only the copula from the first tree level
    fig3 = cvine.plotCopula(depth=0, empirical=True)
    # this should give 4 copula plots
    fig3.show()

Throw away point-wise data:

    # not necissary, but nice to clear out pointwise data
    # when it is no longer needed
    cvine.destroy_simulated_data()
    cvine.destroy_input_data()

Print C-vine info:

    # print detailed info
    cvine.info()

Save the C-vine to file for later sampling:

    cvine.saveVine('cvine_ex.h5')

Reload the saved vine:

    newvine = sv.cvine()
    newvine.read_h5('cvine_ex.h5')

Sample the reclaimed C-vine:

    newvine.simulate(n=1000)
    # plot only the copula from the first tree level
    fig4 = newvine.plotCopula(depth=0, empirical=False)
    # fig4 should be identical to fig3 minus the empirical data
    fig4.show()


License
========

StarVine is distributed under the BSD 3 clause license.

A copy of the license should have been distributed with the StarVine source code.
If not, see [opensource.org](https://opensource.org/licenses/BSD-3-Clause)
