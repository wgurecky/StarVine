##
# @brief Gamma model distribution
from __future__ import print_function, division
from uv_base import UVmodel
import numpy as np
from scipy.special import gamma


class UVGamma(UVmodel):
    """!
    @brief Custom 2 parameter gamma distribution.

    Note:
        Included example.
    """
    def __init__(self, *args, **kwargs):
        # supply parameter string and support range to base class
        # Gamma PDF is supported on (0, +\infty)
        super(UVGamma, self).__init__(paramsString="a, b",
                                      momtype=0,
                                      bounds=[0, float('inf')],
                                      name=kwargs.pop("name", "custom_gamma"))

    def _pdf(self, x, *args):
        """!
        @brief Gamma PDF
        """
        a, b, x = np.ravel(args[0]), np.ravel(args[1]), np.ravel(x)
        return (b ** a) * (x ** (a - 1.0)) * np.exp(-x * b) / gamma(a)

    def _pCheck(self, params):
        """
        @brief Parameter bounds check
        """
        for p in params:
            if p <= 0:
                return False
        return True


def example():
    # Test gamma fit with mle and gmm from
    # from https://github.com/josef-pkt/misc/blob/master/notebooks/ex_gmm_gamma.ipynb
    tstData = \
        np.array([20.5, 31.5, 47.7, 26.2, 44.0, 8.28, 30.8, 17.2, 19.9, 9.96,
                  55.8, 25.2, 29.0, 85.5, 15.1, 28.5, 21.4, 17.7, 6.42, 84.9])
    gamma_model = UVGamma(name='custom_gamma')
    params0 = [2.0, 0.1]

    # ------------------------------------------------------------------------ #
    # Scipy's MLE estimate (fixed location and scale)
    fitted_params = gamma_model.fit(tstData, *params0, floc=0, fscale=1)
    print("---- Scipy MLE params ----")
    print(fitted_params)

    # ------------------------------------------------------------------------ #
    # Custom MLE estimate
    cus_fitted_params = gamma_model.fitMLE(tstData, params0)
    print("---- Custom MLE params ----")
    print(cus_fitted_params)

    # ------------------------------------------------------------------------ #
    # GMM estimate
    gamma_model.setupGMM(tstData, nMoM=4)
    gmm_params = gamma_model.internalGMM.fit(params0, maxiter=4, optim_method='nm', wargs=dict(centered=False))
    print("---- GMM params ----")
    print(gmm_params.params)

    # ------------------------------------------------------------------------ #
    # MCMC estimate
    # Define prior distributions for the paramers in the model
    def logPrior(theta):
        # Define ln(P(model))
        # Assume a and b are uncorrelated ln(P(a,b)) = ln(P(a))*ln(P(b))
        # Assume flat prior: ln(P(a)) == ln(1.0) == ln(P(b))
        a, b = theta
        if a <= 0 or b <= 0:
            # impossible condition.  Return ln(0.0)
            return -np.inf
        return 0.0  # ln(1.0) == 0.0
    gamma_model.setLogPrior(logPrior)
    # Set inital position of all walkers
    gamma_model.setupMCMC(20, params0, tstData)
    gamma_model.fitMCMC(1000)
    # Get results
    samples = gamma_model.sampler.chain[:, 50:, :].reshape((-1, 2))
    print("---- MCMC params ----")
    print("Averge: " + str(np.average(samples, axis=0)) + " +/-sigma :" + str(np.std(samples, axis=0)))

if __name__ == "__main__":
    example()
