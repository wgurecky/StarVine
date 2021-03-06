##
# \brief Multi-variate data container composed of univariate data sets.
from six import iteritems
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
# internal imports
from starvine.uvar.uvd import Uvd
from mv_plot import matrixPairPlot as mpp


class Mvd(object):
    """!
    @brief  Multi-variate data class.
    Performs principal component analysis to reduce the dimensionality
    of a large data set.
    """
    def __init__(self, mvdData=pd.DataFrame(), mvdWeights=pd.DataFrame()):
        self.uvdPool = {}  # Univariate data model storage
        # TODO: accept single pandas data frame with datagroup: "weights"
        self.mvdData = mvdData
        self.mvdDataWeights = mvdWeights
        self.nDims = self.mvdData.shape

    def setData(self, dataDict, weights=None):
        """!
        @brief Collect data from dictionary with {<b>str</b>: <b>np_1darray</b>}
        {key, value} pairs into a pandas dataFrame
        """
        self.mvdData, self.mvdDataWeights = pd.DataFrame(), pd.DataFrame()
        for dataName, data in iteritems(dataDict):
            self.uvdPool[dataName] = Uvd(data, dataName=dataName)
            # TODO: check all datasets are of equal length
            self.mvdData[dataName] = pd.Series(data)
        if weights is None:
            self.mvdDataWeights = \
                pd.DataFrame(np.ones(self.mvdData.shape[0]))
        else:
            self.mvdDataWeights = pd.DataFrame(weights)
        self.nDims = self.mvdData.shape

    def plot(self, **kwargs):
        """!
        @brief generate pairwise scatter plots
        """
        mpp(self.mvdData, self.mvdDataWeights, **kwargs)

    def setUVD(self, uvdList):
        """!
        @brief  Collect uni-variate data sets into a multivariate data object.
        @param  uvdList <b>list</b> of <b>uvar.Uvd</b> instances
        """
        for i, uvd in enumerate(uvdList):
            dataName = uvd.dataName if uvd.dataName is not None else i
            self.uvdPool[dataName] = uvd
            self.mvdData[dataName] = pd.Series(uvd.fieldData)
            self.mvdDataWeights[dataName] = pd.Series(uvd.dataWeights)
        if self.mvdDataWeights is None:
            self.mvdDataWeights = \
                pd.DataFrame(np.ones(self.mvdData.values.shape))
        self.nDims = self.mvdData.shape

    def computeKDEpdf(self, bandwidth=None):
        """!
        @brief Computes mulitvariate kernel density function.  A kernel density function is
        constructed by combining many locally supported "mini PDFs".  This is
        a data smoothing operation.
        @param <b>float</b> (optional) bandwidth.  Default is to use the "scott" factor:
        \f$n^{\frac{-1}{d+4}}\f$
        """
        self.mvdKDEpdf = gaussian_kde(self.mvdData.values, bw_method=bandwidth)
        return self.mvdKDEpdf

    def computeCov(self, weighted=False):
        """!
        @brief computes cov matrix
        @param weighted <b>bool</b> if True, utilizes mvdDataWeights (vols or areas)
            as frequency weights, essentially counting samples which represent more
            "area" or "volume" in the domain more times. True by default.
        """
        if weighted:
            self.mvdCov = np.cov(self.mvdData.values,
                                 aweights=self.mvdDataWeights.values)
        else:
            self.mvdCov = np.cov(self.mvdData.values)
        return self.mvdCov

    def computePC(self):
        """!
        @brief Computes principal components of multivariate data set.
        Provides a measure of explained varience per principal component,
        the principal compenent directions and magnitudes.
        @return <b>list</b>
            [eigen_pairs, frac explained variance, cummulative_explained_var]
        """
        # shift data so that mean ==0 and var==1 on all axes
        std_mvdData = StandardScaler().fit_transform(self.mvdData.values)
        # compute cov matrix
        std_cov = np.cov(std_mvdData.values.T, aweights=self.mvdDataWeights.values)
        eig_vals, eig_vecs = np.linalg.eig(std_cov)
        # u, s, v = np.linalg.svd(std_mvdData.T)
        # Create sorted eigen-pairs
        self.eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_sum = np.sum(eig_vals)
        self.frac_explained_var = [(s / eig_sum) for s in sorted(eig_vals, reverse=True)]
        self.cum_frac_explained_var = np.cumsum(self.frac_explained_var)
        return self.eig_pairs, self.frac_explained_var, self.cum_frac_explained_var

    def computePCOP(self, retainFracVar=0.95, reducedDim=None):
        """!
        @brief Computes a projection matrix.
        Maps from the original data space to a reduced param space.
        @param retainFracVar <b>double</b> Desired fraction of explained varience to retain
        @param reducedDim <b>int</b> Target reduced data dimension
        @return <b>np_ndarray</b> PC projection matrix
        """
        if not hasattr(self, "eig_pairs"):
            self.computePC()
        if not reducedDim:
            reducedDim = self.nDims[1]
        elif (reducedDim < self.nDims[1]) and (reducedDim > 0):
            pass
        else:
            print("ERROR: reducedDim should be <= original data dimension and be >0")
            raise IndexError
        retained_eig_vecs = []
        for i in range(reducedDim):
            retained_eig_vecs.append(self.eig_pairs[i][1].reshape(self.nDims[1], 1))
            if self.cum_frac_explained_var[i] >= retainFracVar:
                break
        self.pcW = np.hstack(retained_eig_vecs)
        return self.pcW

    def applyPCA(self):
        """!
        @brief Provides a reduced order view of the current MVD object.

        @returns (Mvd object, eigenValues_cov, eigenVectors_cov, W)
        Where W is the transformation matrix.
        """
        pass

    def plotExplainedVar(self):
        """!
        @brief Plots fractional explained varience as a function of
        number of principal components retained.
        """
        pass
