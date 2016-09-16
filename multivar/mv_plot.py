##
# \brief Plotting functions supporting multivariate data class.
from scipy.stats import kendalltau
import seaborn as sns

def pairPlotMatrix(self, mvdData, mvdWeights=None, **kwargs):
    """!
    @brief Generates pairwise scatter plots of multi-variate data.
    @brief mvdData  <dataFrame> Multivariate data frame
    @brief mvdWeights  <dataFrame> Data weights
    @param kwargs  optional extra arguments to pass to plotting function
    @return list of matplotlib plt objects
    """
    return sns.pairplot(
        mvdData,
        hist_kwds={'weights': mvdWeights},
        plot_kwds={'size': mvdWeights,
                   'stat_func': kendalltau},
        )

def explainedVarPlot(self, **kwargs):
    """!
    @brief Generates explained varience plot from PCA results.
    """
    pass
