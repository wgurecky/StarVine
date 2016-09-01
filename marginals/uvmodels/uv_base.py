from scipy.stats import rv_continuous
from statmodels.sandbox.regression.gmm import GMM
import numpy as np


class BVmodel(rv_continuous):
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

    Example 2 parameter custom gamma distribution:
        from uv_base import BVmodel
        gamma_model = BVmodel(name='custom_gamma', shapes="a, b")
        fitted_params, loc, scale = gamma_model.fit(xdata)
    """
    def __init__(self, *args, **kwargs):
        pass

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
        self.nMomentConditions = nMoM
        # Require number of moment conditions
        if not self.nMomentConditions:
            raise NotImplementedError
        nobs = data.shape[0]
        exog = np.ones((nobs, self.nMomentConditions))
        self.internalGMM = GMMgeneric(data, exog, None,
                                      weights=w, internal=self, nMoM=self.nMomentConditions)

    def _pdf(self, x, *args):
        """!
        @brief Pure virtual pdf model function
        """
        raise NotImplementedError


class GMMgeneric(GMM):
    """!
    @brief General method of moments base class
    """
    def __init__(self, *args, **kwargs):
        """!
        @brief General method of moments base class
        args[0] endog AKA Y var "response var"
        args[1] exog AKA X var "explan var"
        """
        self.internal = kwargs.pop("internal", None)
        self.weights = kwargs.pop("weights", 1.0)
        self.nMomentConditions = kwargs.pop("nMoM", 2)
        kwargs.setdefault('k_moms', 2)  # moment conditions
        kwargs.setdefault('k_params', 2)  # number model params
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
            error_gn.append(weights * (endog ** n - self.internal.moment(n, params))
                            / np.sum(weights))
        g = np.column_stack(tuple(error_gn))
        return g
