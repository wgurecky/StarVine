##
# \brief Multi-variate data container composed of univariate data sets.

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import mv_plot as mvp

class Mvd(object):
    """!
    @brief  Multi-variate data class.
    Performs principal component analysis as a preprocessing step
    to reduce the dimensionality of a large data set.
    """
    def __init__(self):
        """! @brief Storage for constitutive uni-variate data sets """
        self.uvdPool = {}
        self.mvdData = pd.DataFrame()
        self.mvdDataWeights = pd.DataFrame()

    def setUVD(self, uvdList):
        """!
        @brief  Add uni-variate data sets to multivariate data object.
        @param  uvdList <list> of <starStats.starStatUVD.SSuvd> instances
        """
        for uvd in uvdList:
            self.uvdPool[uvd.dataName] = uvd
            self.mvdData[uvd.dataName]
            # set col dset names
            #
            self.mvdDataWeights[uvd.dataName]
        # check all uvd data sets are of equal length
        for uvdData in self.uvdPool.values():
            assert(self.uvdPool.values()[0][:].shape == uvdData[:].shape)
        self.nDims = (len(self.uvdPool.values()[0][:]), len(self.uvdPool))

    def plot(self, **kwargs):
        """!
        @brief generate pairwise scatter plots
        Ex:
        >>> self.plot(savefig='outfig.png')
        """
        mvp.matrixPairPlot(self.mvdData, **kwargs)

    def computeUVDMoments(self, maxMoment=6):
        """!
        @brief Compute all moments of all uvd's
        @param maxMoment <int> maximum moment to compute (ex: maxMoment=4
        would compute up the the "kurtosis".)
        """
        for uvD in self.uvdPool.values():
            uvD.computeMoments(maxMoment)

    def computeKDEpdf(self, bandwidth=None):
        """!
        @brief Computes mulitvariate kernel density function.  A kernel density function is
        constructed by combining many locally supported "mini PDFs".  This is
        a data smoothing operation - it may be useful as a data preprocessor.
        @param <float> (optional) bandwidth.  Default is to use the "scott" factor:
        \f$n^{\frac{-1}{d+4}}$\f
        """
        self.mvdKDEpdf = gaussian_kde(self.mvdData.values, bw_method=bandwidth)
        return self.mvdKDEpdf

    def computeCov(self, weighted=True):
        """!
        @brief computes cov matrix
        @param weighted <bool> if True, utilizes mvdDataWeights (vols or areas)
            as frequency weights, essentially counting samples which represent more
            "area" or "volume" in the domain more times. True by default.
        """
        if weighted:
            self.mvdCov = np.cov(self.mvdData.values,
                                 aweights=self.mvdDataWeights.values)
        else:
            self.mvdCov = np.cov(self.mvdData.values)

    def computePC(self):
        """!
        @brief Computes principal components of multivariate data set.
        Provides a measure of explained varience per principal component,
        the principal compenent directions and magnitudes.
        @return <list>
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
        @param retainFracVar <double> Desired fraction of explained varience to retain
        @param reducedDim <int> Target reduced data dimension
        @return <np_ndarray> PC projection matrix
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
