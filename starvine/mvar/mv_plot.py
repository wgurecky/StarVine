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
    pair_plot = sns.PairGrid(data, palette=["red"], size=4)
    # UPPER
    #pair_plot.map_upper(sns.kdeplot, cmap="Blues_d")
    pair_plot.map_upper(sns.regplot)
    pair_plot.map_upper(xy_slope)
    #
    # LOWER
    pair_plot.map_lower(plt.scatter, s=23.0/np.log(data.shape[0]))
    pair_plot.map_lower(corrfunc, cstat=corr_stat)
    #
    # DIAG
    # pair_plot.map_diag(sns.distplot, kde=True, norm_hist=True)
    pair_plot.map_diag(plt.hist, edgecolor="white")
    #
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    outfile = kwargs.pop("savefig", None)
    if outfile:
        pair_plot.savefig(outfile)
    # plt.close()
    return pair_plot


def xy_slope(x, y, **kws):
    p, v = np.polyfit(x, y, 1, cov=True)
    slope, intercept = p[0], p[1]
    # r_squared = v[0, 1] ** 2
    ax = plt.gca()
    ax.annotate("slope= {:.3f}".format(slope),
                xy=(0.1, 0.95), xycoords=ax.transAxes)
    ax.annotate("int= {:.3f}".format(intercept),
                xy=(0.1, 0.92), xycoords=ax.transAxes)


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
                    xy=(0.1, 0.95), xycoords=ax.transAxes)
    if cstat is "pearsonr":
        ax.annotate("PsRho= {:.2f}".format(r),
                    xy=(0.1, 0.95), xycoords=ax.transAxes)
    if cstat is "spearmanr":
        ax.annotate("SprRho= {:.2f}".format(r),
                    xy=(0.1, 0.95), xycoords=ax.transAxes)


def explainedVarPlot(self, **kwargs):
    """!
    @brief Generates explained varience plot from PCA results.
    """
    pass
