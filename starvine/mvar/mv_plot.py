##
# \brief Plotting functions supporting multivariate data class.
from scipy.stats import kendalltau, spearmanr, pearsonr, linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def matrixPairPlot(data, weights=None, corr_stat="kendalltau", **kwargs):
    """!
    @brief Plots a matrix of pair plots.
    @param data <pandas dataframe> nDim data set
    @param weights np_1darray of weights to assign to each row in data
    @param corr_stat (optional) correlation statistic for plot
    """
    upper_kde = kwargs.pop("kde", False)
    pair_plot = sns.PairGrid(data, palette=["red"], size=kwargs.pop("size", 5))
    # UPPER
    if upper_kde:
        pair_plot.map_upper(sns.kdeplot, cmap="Blues_d")
    else:
        pair_plot.map_upper(sns.regplot, scatter_kws={'s': 3.0})
        pair_plot.map_upper(xy_slope)
    #
    # LOWER
    if weights is not None:
        weightArray = weights.flatten()
    else:
        weightArray = np.ones(data.shape[0])
    meanWeight = np.mean(weightArray)
    # pt_size = 25. * np.log(len(weightArray)) * (weightArray - np.min(weightArray)) / (np.max(weightArray) + 0.01 - np.min(weightArray)) + 0.2
    pt_size = 5.0 / np.log(data.shape[0]) * weightArray / meanWeight
    pt_size = np.clip(pt_size, 0.1, 50.)
    pair_plot.map_lower(plt.scatter, s=pt_size)
    pair_plot.map_lower(corrfunc, cstat=corr_stat)
    #
    # DIAG
    # pair_plot.map_diag(sns.distplot, kde=True, norm_hist=True, hist_kws={'weights': weightArray})
    pair_plot.map_diag(plt.hist, edgecolor="white", weights=weightArray)
    #
    # set Y axes
    # pair_plot.axes[-1, 0].set_ylim((0.096, 0.104))
    pair_plot.axes[-1, 0].autoscale(tight=True)
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    outfile = kwargs.pop("savefig", None)
    if outfile:
        pair_plot.savefig(outfile)
    # plt.close()
    return pair_plot


def xy_slope(x, y, **kws):
    slope, intercept, r_squared, p, s = linregress(x, y)
    ax = plt.gca()
    ax.annotate("slp= {:.3e}".format(slope),
                xy=(0.05, 0.95), xycoords=ax.transAxes)
    ax.annotate("y0= {:.3e}".format(intercept),
                xy=(0.05, 0.895), xycoords=ax.transAxes)
    ax.annotate("R^2= {:.2f}".format(r_squared),
                xy=(0.75, 0.95), xycoords=ax.transAxes)


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
                    xy=(0.05, 0.95), xycoords=ax.transAxes)
    if cstat is "pearsonr":
        ax.annotate("PsRho= {:.2f}".format(r),
                    xy=(0.05, 0.95), xycoords=ax.transAxes)
    if cstat is "spearmanr":
        ax.annotate("SprRho= {:.2f}".format(r),
                    xy=(0.05, 0.95), xycoords=ax.transAxes)


def explainedVarPlot(self, **kwargs):
    """!
    @brief Generates explained varience plot from PCA results.
    """
    pass
