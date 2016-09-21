import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy import integrate
from statmodels.sandbox.regression.gmm import GMM


class UVmodel(GMM):
    """!
    @brief Univariate paramaterized model base class.
    Meant as abstract base class.
    """
    def __init__(self, *args, **kwargs):
        self.nParams = 0
        self.params = None
        self.xdata = None
        self.nMomentConditions = None

    def evalPDF(self, x, params=None):
        """!
        @brief Eval model pdf. Pure virtual method
        @param x  <np_ndarray> or <float>.  Evaluate model pdf at [x]
        """
        raise NotImplementedError

    def evalCDF(self, x, params=None):
        """!
        @brief Eval model cdf.  Virtual method
        default numerical implementation.  If an analytic CDF is
        avalible override this method in the subclass.
        @param x  <np_ndarray> or <float>.  Evaluate model cdf at [x]
        """
        return integrate.quad(self.evalPDF, a=0, b=x, args=params)

    def evalInvCDF(self, p, params=None):
        """!
        @brief Inverse of CDF. Virtual method
        default numerical implementation. If an analytic invCDF is
        available override this method in the subclass.
        """
        res = minimize(lambda x: np.sqrt((self.evalCDF(x) - p) ** 2),
                       0.5,
                       constraints=({'type': 'ineq', 'fun': lambda x: x},),
                       tol=1e-9,
                       method='SLSQP')
        return res.x

    def setParams(self, newParams):
        """!
        @brief set model parameters (initial guess)
        @param newParams <np_array>
        """
        for i, par in enumerate(newParams):
            if i > len(self.nParams):
                print("WARNING: parameter length mismatch")
                break
            else:
                self.params[i] = par

    def getParams(self):
        """!
        @brief Return model parameters
        """
        return self.params

    def getModelInfo(self):
        """!
        @brief  Return model and parameter description.
        """
        pass

    def fitHistogram(self, xdata, weights=None, **kwargs):
        binValues, binBounds = np.histogram(self.fieldData, bins="auto",
                                            weights=self.scaledWeights, density=True)
        return binValues, binBounds

    def fitModelKDE(self, xdata, **kwargs):
        """!
        @brief Compute kernel density estimate of PDF.
        Uses a linear combination of mini gaussian functions
        to construct a smooth fit.
        """
        return gaussian_kde(xdata, bw_method=None)

    def fitModelLS(self, xdata, weights=None, **kwargs):
        """!
        @brief Fit model to data using least squares
        @param xdata <np_array>
        @param weights <np_array> (optional), default is 1.0
        """
        if not weights:
            weights = 1.0

        def costFn(fit_params):
            E = np.linalg.norm(weights * (self.fitModelKDE(xdata) - self.evalPDF(xdata, fit_params)))
            return E  # L2 error
        opti_fit = \
            minimize(costFn,
                     self.getParams(),
                     constraints=kwargs.pop('constraints', ()),
                     tol=1e-6,
                     method='SLSQP',
                     **kwargs)
        self.params = opti_fit.x

    def fitModelGMM(self, xdata, weights=None, **kwargs):
        """!
        @brief Fit model to data using general method of moments.  Must be able
        to evaluate model moments either analytically or numerically.
        Model moments must be implemented in a `momcond` method

        @param xdata <np_array>
        @param weights <np_array> (optional)
        """
        kwargs.setdefault('k_moms', self.nMomentConditions)
        kwargs.setdefault('k_params', len(self.params))
        # call __init__ of GMM base class.  Use "dummy" exog and instrument data
        super(UVmodel, self).__init__(xdata,
                                      np.ones((len(xdata), self.nMomentConditions)),
                                      None,
                                      **kwargs)
        # Perform 2 step GMM fit
        gmmResult = \
            self.fit(self.params, maxiter=2,
                     optim_method='slsqp', wargs=dict(centered=False))
        print(gmmResult.summary())
        return gmmResult

    def setMomentConds(self, momentConds):
        self.g = momentConds
        self.nMomentConditions = np.shape(self.g)[0]

    def momcond(self, params):
        """!
        @brief Moment conditions for GMM
        """
        raise NotImplementedError

    def fitModelMCMC(self, xdata, weights=None, **kwargs):
        """!
        @brief Fit model to data using marcov chain monte carlo
        @param xdata <np_array>
        @param weights <np_array> (optional)
        """
        # TODO: MCMC parameter estimation
        pass

    def plotHistogram(self):
        pass

    def plotKDE(self):
        pass

    def plotModel(self):
        pass
