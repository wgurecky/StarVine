##
# \brief Base vine class
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import networkx as nx
import numpy as np
import pandas as pd
from six import iteritems
# from starvine.mvar.mv_plot import matrixPairPlot


class BaseVine(object):
    """!
    @brief Regular vine base class.
    """
    def __init__(self, data, dataWeights=None, **kwargs):
        self.trial_copula_dict = \
                self._validate_trial_copula( \
                kwargs.get("trial_copula", self._all_trial_copula))
        self.data = data
        self.weights = dataWeights

    def _validate_trial_copula(self, trial_copula):
        assert isinstance(trial_copula, dict)
        for key, val in iteritems(trial_copula):
            assert key in self._all_trial_copula
            assert self._all_trial_copula[key] == val

    @property
    def _all_trial_copula(self):
        default_copula = {'t': 0,
                          'gauss': 0,
                          'frank': 0,
                          'frank-90': 1,
                          'frank-180': 2,
                          'frank-270': 3,
                          'clayton': 0,
                          'clayton-90': 1,
                          'clayton-180': 2,
                          'clayton-270': 3,
                          'gumbel': 0,
                          'gumbel-90': 1,
                          'gumbel-180': 2,
                          'gumbel-270': 3,
                         }
        return default_copula

    def loadVineStructure(self, vS):
        """!
        @brief Load saved vine structure
        """
        pass

    def vineNLLH(self, vineParams=[None], **kwargs):
        """!
        @brief Compute the vine negative log likelihood.  Used for
        simulatneous MLE estimation of PCC model parameters.
        Loops through all tree levels and sums all NLL.
        @param vineParams <b>np_array</b>  Flattened array of all copula parametrs in vine
        """
        if not any(vineParams):
            self._initVineParams()
        self.nLLH = 0.
        for lvl, tree in enumerate(self.vine):
            treeNLLH = tree.treeNLLH(vineParams[self.vineParamsMap[lvl]:
                                                self.vineParamsMap[lvl + 1]])
            self.nLLH += treeNLLH
        return self.nLLH

    def _initVineParams(self):
        self.vineParams = []
        self.vineParamsMap = [0]
        for lvl, tree in enumerate(self.vine):
            self.vineParams.append(tree._initTreeParamMap())
            self.vineParamsMap.append(self.vineParamsMap[lvl] + len(self.vineParams[lvl]))

    def sfitMLE(self, **kwargs):
        """!
        @brief Simulataneously estimate all copula paramters in the
        vine by MLE.  Uses SLSQP method by default.
        """
        self._initVineParams()
        params0 = np.array(self.vineParams).flatten()
        self.fittedParams = minimize(self.vineNLLH, params0, args=(),
                                     method=kwargs.pop("method", "SLSQP"),
                                     tol=kwargs.pop("tol", 1e-5))

    def treeHfun(self, level=0):
        """!
        @brief Operates on a tree, T_(i).
        The conditional distribution is evaluated
        at each edge in the tree providing univariate distributions that
        populate the dataFrame in the tree level T_(i+1)
        """
        pass

    def sample(self, n=1000):
        """!
        @brief Draws n samples from the vine.
        @param n int. number of samples to draw
        @returns  size == (n, nvars) <b>pandas.DataFrame</b>
            samples from vine
        """
        # gen random samples
        u_n0 = np.random.rand(n)
        u_n1 = np.random.rand(n)

        # obtain edge from last tree in vine
        current_tree = self.vine[-1]
        n0, n1 = list(current_tree.tree.edges())[0]
        edge_info = current_tree.tree[n0][n1]

        # sample from edge of last tree
        u_n1 = edge_info["hinv-dist"](u_n0, u_n1)
        edge_sample = {n0: u_n0, n1: u_n1}
        # matrixPairPlot(pd.DataFrame(edge_sample), savefig="c_test/tree_1_edge_sample.png")

        # store edge sample inside graph data struct
        current_tree.tree[n0][n1]['sample'] = edge_sample

        # Three nodes in the above tree that contributed to the
        # construction of this edge.
        # Node labels according to  (prev_n0|prev_n2), (prev_n1|prev_n2)
        # Only applicable if the vine has atleast 2 levels
        if len(self.vine) > 1:
            prev_n0, prev_n1, prev_n2 = edge_info['one-fold']

            ## \brief Entrance to starvine.vine.tree.Vtree._sampleEdge()
            current_tree._sampleEdge(prev_n0, prev_n2, n0, n1, n, self.vine)
            current_tree._sampleEdge(prev_n1, prev_n2, n0, n1, n, self.vine)

        sample_result = {}
        tree_0 = self.vine[0].tree
        for edge in tree_0.edges():
            n0, n1 = edge
            edge_info = tree_0[n0][n1]
            if not n0 in list(sample_result.keys()):
                sample_result[n0] = edge_info['sample'][n0]
            if not n1 in list(sample_result.keys()):
                sample_result[n1] = edge_info['sample'][n1]

        # clean up
        for base_tree in self.vine:
            for edge in base_tree.tree.edges():
                n0, n1 = edge
                base_tree.tree[n0][n1].pop('sample')

        # convert sample dict of arrays to dataFrame
        return pd.DataFrame(sample_result)

    def sampleScale(self, n, frozen_margin_dict):
        """!
        @brief Sample vine copula and apply inverse transform sampling
            to margins.
        @param n int. number of samples to draw.
        @param frozen_margin_dict dict of frozen single dimensional
            prob density functions. See: scipy.stats.rv_continuous
        """
        df_x = self.sample(n)
        return self.scaleSamples(df_x, frozen_margin_dict)

    def scaleSamples(self, df_x, frozen_margin_dict):
        """!
        @brief Apply inverse transform sampling
        @param df_x corrolated samples in [0, 1] from vine copula
        @param frozen_margin_dict dict of frozen single dimensional
            prob density functions. See: scipy.stats.rv_continuous.
            Each entry in the dict is an instance of:
                frozen_marginal_model  frozen scipy.stats.rv_continuous
                python object
            or is a inverse cdf fuction
        """
        sample_result = {}
        assert isinstance(df_x, pd.DataFrame)
        for col_name in df_x.columns:
            assert col_name in frozen_margin_dict.keys()
            frozen_marginal_model = frozen_margin_dict[col_name]
            x = df_x[col_name]
            if hasattr(frozen_marginal_model, 'ppf'):
                sample_result[col_name] = frozen_marginal_model.ppf(x)
            else:
                sample_result[col_name] = frozen_marginal_model(x)
        return pd.DataFrame(sample_result)

    def plotVine(self, plotAll=True, savefig=None):
        """!
        @brief Plots the vine's graph structure.
        @param plotAll (optional) Plot the entire vine structure
        @param savefig (optional) filename of output image.
        """
        plt.figure(10, figsize=(6 + 0.3 * self.nLevels, 3 * self.nLevels))
        for i, treeL in enumerate(self.vine):
            plt.subplot(self.nLevels, 1, i + 1)
            plt.title("Tree Level: %d" % i)
            pos = nx.spring_layout(treeL.tree)
            nx.draw(treeL.tree, pos, with_labels=True, font_size=10, font_weight="bold")
            # specifiy edge labels explicitly
            edge_labels = dict([((u, v,), round(d['weight'], 2))
                                for u, v, d in treeL.tree.edges(data=True)])
            nx.draw_networkx_edge_labels(treeL.tree, pos, edge_labels=edge_labels)
        if savefig is not None:
            plt.savefig(savefig)
        plt.close(10)
