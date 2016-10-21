##
# \brief Plotting functions supporting multivariate data class.
from scipy.stats import kendalltau, spearmanr, pearsonr, linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def matrixPairPlot(data, weights, corr_stat="kendalltau", **kwargs):
    """!
    @brief Plots a matrix of pair plots.
    @param data <pandas dataframe> nDim data set
    @param corr_stat (optional) correlation statistic for plot
    """
    upper_kde = kwargs.pop("kde", False)
    pair_plot = sns.PairGrid(data, palette=["red"], size=4)
    # UPPER
    if upper_kde:
        pair_plot.map_upper(sns.kdeplot, cmap="Blues_d")
    else:
        pair_plot.map_upper(sns.regplot, scatter_kws={'s': 3.0})
        pair_plot.map_upper(xy_slope)
    #
    # LOWER
    weightArray = weights.values.flatten()
    meanWeight = np.mean(weightArray)
    pair_plot.map_lower(plt.scatter, s=25.0/np.log(data.shape[0]) * weightArray / meanWeight )
    pair_plot.map_lower(corrfunc, cstat=corr_stat)
    #
    # DIAG
    # pair_plot.map_diag(sns.distplot, kde=True, norm_hist=True, hist_kws={'weights': weightArray})
    pair_plot.map_diag(plt.hist, edgecolor="white", weights=weightArray, bins=20)
    #
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
