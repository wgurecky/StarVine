##
# \brief Bivariate plotting functions.
# Depends on the seaborn python package for simplified
# bivariate plotting.
from scipy.stats import kendalltau, spearmanr, pearsonr
import seaborn as sns


def bvPairPlot(u, v, corr_stat="kendalltau", **kwargs):
    stat_funcs = {"kendalltau": kendalltau,
                  "spearmanr": spearmanr,
                  "pearsonr": pearsonr}
    pair_plot = sns.jointplot(u, v, stat_func=stat_funcs[corr_stat])
    outfile = kwargs.pop("savefig", None)
    if outfile:
        pair_plot.savefig(outfile)
    return pair_plot
