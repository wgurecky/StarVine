import numpy as np
from scipy.optimize import minimize
from statmodels.sandbox.regression.gmm import GMM


class UVmodel(GMM):
    """!
    @brief Univariate paramaterized model base class

    All model distributions need:
        - storage for model parameters
        - supply initial guess for parameters
        - evaluate the model pdf (sample pdf)
        - evaluate the cdf
    """
    def __init__(self, params=[], *args, **kwargs):
        self.params = params
        self.x = None

    def evalPDF(self, x, params=None):
        """!
        @brief Eval model pdf. Virtual method
        @param x  <np_ndarray> or <float>.  Evaluate model pdf at [x]
        """
        pass

    def evalCDF(self, x, params=None):
        """!
        @brief Eval model cdf.  Virtual method
        @param x  <np_ndarray> or <float>.  Evaluate model cdf at [x]
        """
        pass

    def setParams(self, newParams):
        """!
        @brief set model parameters (initial guess)
        @param newParams <np_array>
        """
        for i, par in enumerate(newParams):
            self.params[i] = par
            if i > len(self.params):
                print("WARNING: parameter length mismatch")
                break

    def getParams(self):
        return self.params

    def fitModelLS(self, xdata, ydata, weights=None, **kwargs):
        """!
        @brief Fit model to data using least squares
        @param xdata <np_array>
        @param ydata <np_array>
        @param weights <np_array> (optional)
        """
        if not weights:
            weights = 1.0

        def costFn(fit_params):
            E = np.linalg.norm(weights * (self.ydata - self.evalPDF(xdata, fit_params)))
            return E  # L2 error
        opti_fit = \
            minimize(costFn,
                     self.getParams(),
                     constraints=kwargs.pop('constraints', ()),
                     tol=1e-6,
                     method='SLSQP')
        self.params = opti_fit.x

    def fitModelGMM(self, xdata, ydata, weights=None, **kwargs):
        """!
        @brief Fit model to data using general method of moments.  Must be able
        to evaluate model moments either analytically or numerically.
        Model moments must be implemented in a `momcond` method

        @param xdata <np_array>
        @param ydata <np_array>
        @param weights <np_array> (optional)
        """
        kwargs.setdefault('k_moms', self.nMomentConditions)
        kwargs.setdefault('k_params', len(self.params))
        # call __init__ of GMM base class
        super(UVmodel, self).__init__(ydata, xdata, **kwargs)
        # Perform 2 step GMM fit
        # note GMM.fit requires momcond(model_params) to be implemented
        gmmResult = \
            self.fit(self.params, maxiter=2,
                     optim_method='slsqp', wargs=dict(centered=False))
        print(gmmResult.summary())
        return gmmResult

    def momcond(self, params):
        """!
        @brief Compose moment conditions for GMM
        @param params <np_array> Model parameters
        """
        endog = self.endog  # y var
        # Example: vector valued moment conditions for gamma(p, q) pdf
        g1 = endog - params[0] / params[1]
        g2 = endog**2 - (params[0] / params[1])**2 - (params[0] / params[1]**2)
        g = np.column_stack((g1, g2))
        return g

    def fitModelMCMC(self, xdata, ydata, weights=None, **kwargs):
        """!
        @brief Fit model to data using marcov chain monte carlo
        @param xdata <np_array>
        @param ydata <np_array>
        @param weights <np_array> (optional)
        """
        # TODO: Implement MCMC parameter estimation
        pass
