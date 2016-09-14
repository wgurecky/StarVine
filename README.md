About
========

StarVine provides tools to construct C- and D-vine copula to model
dependence structures in multivariate data sets.

Install
========

To install:

    python setup.py install

To install a development version:

    python setup.py develop

Requires:

    - numpy
    - scipy
    - networkx
    - pandas
    - h5py
    - emcee
    - mpi4py : (optional)

Docs
=====

Documentation is provided in the /doc directory.

build the docs from the doc dir:

    cd doc
    doxygen Doxyfile.in
    make html

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

Throw away point-wise data to free RAM:

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

We can sample the reclaimed C-vine:

    newvine.simulate(n=1000)
    # plot only the copula from the first tree level
    fig4 = newvine.plotCopula(depth=0, empirical=False)
    # fig4 should be identical to fig3 minus the empirical data
    fig4.show()


License
========

BSD 3 clause

A copy of the license should have been distributed with this source.
