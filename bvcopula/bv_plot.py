##
# \brief Bivariate plotting functions.
# Depends on the seaborn python package for simplified
# bivariate plotting.
from scipy.stats import kendalltau, spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np


def bvContour(x1, x2, y, **kwargs):
    contour_plot = plt.figure()
    outfile = kwargs.pop("savefig", None)
    sns.interactplot(x1, x2, y, filled=True,
                     scatter_kws={"marker": "x", "markersize": 1},
                     contour_kws={"linewidths": 2},
                     **kwargs)
    if outfile:
        contour_plot.savefig(outfile)
    plt.close()
    return contour_plot


def bvJointPlot(u, v, corr_stat="kendalltau", **kwargs):
    stat_funcs = {"kendalltau": kendalltau,
                  "spearmanr": spearmanr,
                  "pearsonr": pearsonr}
    outfile = kwargs.pop("savefig", None)
    joint_plt = sns.jointplot(u, v, stat_func=stat_funcs[corr_stat], **kwargs)
    if outfile:
        joint_plt.savefig(outfile)
    return joint_plt


def bvPairPlot(u, v, corr_stat="kendalltau", **kwargs):
    data = DataFrame(np.array([u, v]).T, columns=kwargs.pop("labels", None))
    #
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
