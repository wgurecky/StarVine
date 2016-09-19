##
# \brief Plotting functions supporting multivariate data class.
from scipy.stats import kendalltau, spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np


def matrixPairPlot(data, corr_stat="kendalltau", **kwargs):
    """!
    @brief Plots a matrix of pair plots.
    @param data <pandas dataframe> nDim data set
    @param corr_stat (optional) correlation statistic for plot
    """
    pair_plot = sns.PairGrid(data, palette=["red"])
    pair_plot.map_upper(sns.kdeplot, cmap="Blues_d")
    pair_plot.map_diag(sns.distplot, kde=False)
    #
    pair_plot.map_lower(plt.scatter, s=10)
    pair_plot.map_lower(corrfunc, cstat=corr_stat)
    #
    outfile = kwargs.pop("savefig", None)
    if outfile:
        pair_plot.savefig(outfile)
    return pair_plot


def corrfunc(x, y, **kws):
    stat_funcs = {"kendalltau": kendalltau,
                  "spearmanr": spearmanr,
                  "pearsonr": pearsonr}
    cstat = kws.pop("cstat", "kendalltau")
    stat_func = stat_funcs[cstat]
    r, _ = stat_func(x, y)
    ax = plt.gca()
    if cstat is "kendalltau":
        ax.annotate("kTau= {:.2f}".format(r),
                    xy=(0.1, 0.9), xycoords=ax.transAxes)
    if cstat is "pearsonr":
        ax.annotate("PsRho= {:.2f}".format(r),
                    xy=(0.1, 0.9), xycoords=ax.transAxes)
    if cstat is "spearmanr":
        ax.annotate("SprRho= {:.2f}".format(r),
                    xy=(0.1, 0.9), xycoords=ax.transAxes)


def explainedVarPlot(self, **kwargs):
    """!
    @brief Generates explained varience plot from PCA results.
    """
    pass
