from scipy.stats import rv_continuous
from statsmodels.sandbox.regression.gmm import GMM
import numpy as np
from scipy.special import psi


class UVmodel(rv_continuous):
    """!
    @brief Univariate parameterized model base class.
    Extends the capabilities of the rv_continuous base
    class of the scipy.stats module with the GMM capabilites
    of statmodels.sandbox.regression.gmm (general method
    of moments fitting).

    With the combined capability of scipy and statmodels,
    it is easy to obtain maximum likelyhood and MoM
    estimators for univariate model parameters.

    Methods inherited from scipy.stats.rv_continuous:

        rvs(*args, **kwds)  Random variates of given type.
        pdf(x, *args, **kwds)  Probability density function at x of the given RV.
        logpdf(x, *args, **kwds)  Log of the probability density function at x of the given RV.
        cdf(x, *args, **kwds)  Cumulative distribution function of the given RV.
        logcdf(x, *args, **kwds)  Log of the cumulative distribution function at x of the given RV.
        sf(x, *args, **kwds)  Survival function (1 - cdf) at x of the given RV.
        logsf(x, *args, **kwds)  Log of the survival function of the given RV.
        ppf(q, *args, **kwds)  Percent point function (inverse of cdf) at q of the given RV.
        isf(q, *args, **kwds)  Inverse survival function (inverse of sf) at q of the given RV.
        moment(n, *args, **kwds)  n-th order non-central moment of distribution.
        stats(*args, **kwds)  Some statistics of the given RV.
        entropy(*args, **kwds)  Differential entropy of the RV.
        expect([func, args, loc, scale, lb, ub, ...])  Calculate expected value of a
            function with respect to the distribution.
        median(*args, **kwds)  Median of the distribution.
        mean(*args, **kwds)  Mean of the distribution.
        std(*args, **kwds)  Standard deviation of the distribution.
        var(*args, **kwds)  Variance of the distribution.
        interval(alpha, *args, **kwds)  Confidence interval with equal areas around the median.
        __call__(*args, **kwds)  Freeze the distribution for the given arguments.
        fit(data, *args, **kwds)  Return MLEs for shape, location, and scale parameters from data.
        fit_loc_scale(data, *args)  Estimate loc and scale parameters from data using 1st and 2nd moments.
        nnlf(theta, x)  Return negative loglikelihood function.

    """
    def __init__(self, paramsString, momtype, bounds, *args, **kwargs):
        super(UVmodel, self).__init__(shapes=paramsString,
                                      momtype=momtype,
                                      a=bounds[0],
                                      b=bounds[1],
                                      name=kwargs.pop("name", None))
        # Infer number of model params from shapes string (clunky but a necissary evil
        # due to the way rv_continuous is subclassed)
        self.nParams = len(paramsString.split(","))

    def setupGMM(self, data, w=1.0, nMoM=None):
        """!
        @brief General Method of Moments setup.
        @param data Input univariate data
        @param w  Data weights

        Example General Method of Moments use:
            my_uvmodel = UVmodel(...)
            params0 = np.array([2.0, 1.0])
            my_uvmodel.internalGMM.fit(params0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
        """
        self.nMomentConditions = max((nMoM, self.nParams))
        nobs = data.shape[0]
        instruments = exog = np.ones((nobs, self.nMomentConditions))
        self.internalGMM = GMMgeneric(data, exog, instruments,
                                      internal=self,
                                      weights=w,
                                      nMoM=self.nMomentConditions)

    def _pdf(self, x, *args):
        """!
        @brief Pure virtual pdf model function

        Note:
            the number of paramers is "infered" from the signature of
            the _pdf OR _cdf methods.
        """
        raise NotImplementedError

    def _cdf(self, x, *args):
        """!
        @brief Pure virtual cdf model function
        """
        raise NotImplementedError


class GMMgeneric(GMM):
    """!
    @brief General method of moments base class
    """
    def __init__(self, *args, **kwargs):
        """!
        @brief General method of moments base class
        @param endog  Response data
        @param exog   Explanatory data
        @param instruments (optional) defaults to None
        @param internal internal univariate model - must have a moment() method.
        """
        self.internalModel = kwargs.pop("internal", None)
        self.weights = kwargs.pop("weights", 1.0)
        self.nMomentConditions = kwargs.pop("nMoM", self.internalModel.nParams)
        kwargs.setdefault('k_moms', self.nMomentConditions)  # moment conditions
        kwargs.setdefault('k_params', self.internalModel.nParams)  # number model params
        super(GMMgeneric, self).__init__(*args, **kwargs)

    def momcond(self, params):
        """!
        @brief Default moment condition generator implementation.
        @param params Model parameters
        """
        error_gn = []
        weights = self.weights
        endog = self.endog
        for n in range(1, self.nMomentConditions + 1):
            # nth moment condition (error between sample moment and model moment)
            error_gn.append(weights * (endog ** n - self.internalModel.moment(n, *params))
                            / np.sum(weights))
        g = np.column_stack(tuple(error_gn))
        return g
