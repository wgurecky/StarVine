##
# \brief Uni-variate data container
#

import pylab as plt
import numpy as np
from scipy.stats import gaussian_kde


class Uvd(object):
    """!
    @brief Container for univariate data.
    """
    def __init__(self, *args, **kwargs):
        self.dataName = (args[0] if args else None)
        self.fieldData = kwargs.pop("fieldData", None)
        self.dataWeights = kwargs.pop("dataWeights", None)
        self.boundingPlanes = kwargs.pop("boundingPlanes", None)
        self.updateDB(lambda: None)

    def updateDB(self, f):
        """!
        @brief When field data is modified; automatically
        update several basic attributes.
        @param f <function>  Function that modifies self.fieldData
        """
        def wrapped_f(*args, **kwargs):
            return f(*args, **kwargs)
        if self.fieldData:
            self.dataBounds = (np.min(self.fieldData), np.max(self.fieldData))
            self.scaledWeights = (1. / np.sum(self.dataWeights)) * (self.dataWeights)
        return wrapped_f

    def setName(self, dataName):
        self.dataName = dataName

    def setAxialBounds(self, boundingPlanes):
        self.boundingPlanes = boundingPlanes

    @updateDB
    def setData(self, fieldData, dataWeights, boundingPlanes=None):
        """!
        @brief Set the univariate field data and weights
        @param fieldData    Scalar surface or volumetric field data
        @param dataWeights  weights of the data - typically areas or volumes
        @param boundingPlanes (optional) <tuple> of upper and lower axial bounds
        """
        self.fieldData = fieldData
        self.dataWeights = dataWeights
        self.boundingPlanes = boundingPlanes

    def setFrequencyBins(self, binBounds=[]):
        """!
        @brief Set frequency bin structure used when constructing histograms
        @param <list> or np_array of frequency bin bounds
        """
        self.nBins = len(binBounds) - 1
        self.binBounds = np.array(binBounds)

    def computeCentralMoments(self, maxMoment=6):
        """!
        @brief Computes N sample moments about the mean of univariate distribution

        The mean is \f$\frac{1}{N_{samples} w_{tot}}\sum_i^N{X_i * w_i}$\f
        and \f$w_{tot} = \sum{w_i}$\f
        where \f$X_i$\f is the sample value (Temperature_i, TKE_i, .. ect)
        and \f$w_i$\f is the sample weight.

        The nth sample moment about the mean is:
        \f$ \hat{m^n} = \frac{1}{n-1} * \sum{((X_i - \bar{X})^n} $\f

        where N=maxMoment.
        @param maxMoment <int> maximum moment to compute.  Default=6 (hyperflatness)
        """
        dataLen = len(self.dataWeights)
        self.sampleMean = (1 / (dataLen * np.average(self.dataWeights))) * \
            np.sum(self.fieldData * self.dataWeights)
        self.meanMoments = np.zeros(maxMoment)
        for m in range(2, maxMoment):
            self.meanMoments[m] = (1. / (dataLen - m)) * \
                (1. / np.average(self.dataWeights)) ** m * \
                np.sum((self.dataWeights * (self.fieldData - self.sampleMean)) ** m)

    def computeHistogram(self, binBounds=[]):
        """!
        @brief Computes heights of all bins in histogram
        given number of bins and bin widths.
        @note: Automatically computes frequency bin structure if not
        specifed by the user.  Uses "sturges" method for bin structure
        by default.  See numpy.histogram documentation for details.

        @param binBounds (optional) <list> or <np_array> of bin bounds
        @return values, bins
        """
        if not binBounds:
            # np.histogram accepts keyowrd "auto" for bins argument
            binBounds = "auto"
        binValues, binBounds = np.histogram(self.fieldData, bins=binBounds,
                                            weights=self.scaledWeights, density=True)
        self.setFrequencyBins(binBounds)
        return binValues, binBounds

    def computeKDEpdf(self, bandwidth=None):
        """!
        @brief Computes kernel density function.  A kernel density function is
        constructed by combining many locally supported "mini PDFs".  This is
        a data smoothing operation - it may be useful as a data preprocessor.
        @param <float> (optional) bandwidth.  Default is to use the "scott" factor:
        \f$n^{\frac{-1}{d+4}}$\f
        """
        self.kde_pdf = gaussian_kde(self.scaledData, bw_method=bandwidth)

    def computeKDEcdf(self, bandwidth=None):
        """!
        @brief Computes a kernel density estimate of a cumulative univariate
        distribution function.
        @param <float> (optional) bandwith.  Default is to use "scott" factor.
        """
        # ensure self.kde_pdf is initilized
        if not hasattr(self, "kde_pdf"):
            self.computeKDEpdf(bandwidth)
        self.kde_cdf = lambda u: self.kde_pdf.integrate_box([0.0], [u])

    def plotHistogram(self, modelFn=None, outFile="test_uvd.png"):
        """!
        @brief Utility fn that plots univar histogram with optional smooth model fn
        superimposed on the data.
        @param modelFn (optional) function which accepts a vector of [X] and
        returns a 1d_vector of responses [Y].  Ex: use modelFn=computeKDEpdf()
        """
        plt.figure()
        self.computeHistogram()
        n, bins, patches = plt.hist(self.fieldData, bins=self.binBounds,
                                    weights=self.scaledWeights, normed=True)
        plt.xlabel(self.dataName)
        plt.ylabel("Fractional Area Occupied")
        if modelFn is not None:
            yvals = modelFn(bins)
            plt.plot(bins, yvals, 'k--', linewidth=2.0)
        plt.savefig(outFile)

    def computeMOM(self, modelFn=None, modelDomain=[0, 1]):
        """!
        @brief Performs 2 step general method of moments (GMM).
        Finds MoM estimators for model
        parameters which best fit the sample data.

        @note: Transform data domain to [0, 1] by default before
        fitting model distribution.

        @param modelFn (optional function) Canidate model to fit to the
        sample data.  Default assumes beta distribution.
        """
        dataWidth = np.abs(self.dataBounds[1] - dataBounds[0])
        transFieldData = (self.dataBounds - self.dataBounds[0]) / dataWidth
        transFieldData *= np.abs(modelDomain[1] - modelDomain[0])
        # shift to modelDomain
        transFieldData -= (np.min(transFieldData) - modelDomain[0])

        # TODO:
        # perfrom first MoM parameter estimate with Idendity W matrix
        # update W matrix
        # perform second MoM parameter estiamte with new W
        # store best fit model and parameters
        pass
