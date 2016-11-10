##
# \brief Bivariate plotting functions.
# Depends on the seaborn python package for simplified
# bivariate plotting.
from __future__ import print_function, absolute_import
from scipy.stats import kendalltau, spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from pylab import contour, contourf, griddata


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

def bvContourf(x1, x2, z, **kwargs):
    contour_plot = plt.figure()
    outfile = kwargs.pop("savefig", None)
    # create interpolation grid support points
    xx = np.linspace(x1.min(), x1.max(), 150)
    yy = np.linspace(x2.min(), x2.max(), 150)
    # create grid required by pl.contour
    x_grid, y_grid = np.meshgrid(xx, yy)
    # interpolate data to meshgrid
    z_grid = griddata(x1, x2, z, x_grid, y_grid,
                      interp='linear',
                      )
    # plot contour
    contour_plot = plt.figure()
    plt.subplot(1, 1, 1)
    cf = plt.contourf(x_grid, y_grid, z_grid, alpha=0.8, cmap="GnBu")
    cs = plt.contour(x_grid, y_grid, z_grid, 25, colors='k', hold='on', antialiased=True)
    plt.clabel(cs, fontsize=8, inline=1)
    cs = plt.colorbar(cf, shrink=0.8, extend='both', alpha=0.8)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    if outfile:
        contour_plot.savefig(outfile)
    plt.close()
    return contour_plot


def bvJointPlot(u, v, corr_stat="kendalltau", vs=None, **kwargs):
    stat_funcs = {"kendalltau": kendalltau,
                  "spearmanr": spearmanr,
                  "pearsonr": pearsonr}
    outfile = kwargs.pop("savefig", None)
    joint_plt = sns.jointplot(x=u, y=v, stat_func=stat_funcs[corr_stat], **kwargs)
    vsData = vs
    if vsData is not None:
        joint_plt.x, joint_plt.y = vsData[0], vsData[1]
        sb_color = sns.xkcd_palette(["faded green"])[0]
        joint_plt.plot_joint(plt.scatter, s=6, alpha=0.3, c=sb_color, marker='o', edgecolors='face')
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
