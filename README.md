About
========

StarVine provides tools to construct C- and D-vine copula to investigate
dependence structures in multivariate data sets.

Install
========

To install:

    python setup.py develop

Requires:

    - numpy
    - scipy
    - networkx
    - pandas
    - h5py
    - mpi4py

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

Docs
=====

Documentation is provided in the /doc directory.

License
========

BSD 3 clause

Copyright © 2016 William Gurecky
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the organization nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY William Gurecky ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL William Gurecky BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

