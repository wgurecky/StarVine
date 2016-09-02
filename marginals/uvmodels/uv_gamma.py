##
# @brief Gamma model distribution
from uv_base import UVmodel
import numpy as np
from math import gamma


class UVGamma(UVmodel):
    """!
    @brief Custom gamma distribution.

    Example 2 parameter custom gamma distribution:

        from uv_library import UVGamma
        gamma_model = UVGamma(name='custom_gamma')
        fitted_params = gamma_model.fit(xdata)
        print(fitted_params)
        # GMM fit
        gamma_model.setupGMM(xdata)
        params0 = np.array([3.0, 1.0])
        res = gamma_model.internalGMM.fit(params0, maxiter=2, optim_method='nm')
        print(res.params)
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
        Gamma PDF
        """
        a, b = args[0], args[1]
        return (b ** a) * (x ** (a - 1.0)) * np.exp(-x * b) / gamma(a)

def example():
    # Test gamma fit with mle and gmm from
    # from https://github.com/josef-pkt/misc/blob/master/notebooks/ex_gmm_gamma.ipynb
    tstData = \
        np.array([20.5, 31.5, 47.7, 26.2, 44.0, 8.28, 30.8, 17.2, 19.9, 9.96,
                  55.8, 25.2, 29.0, 85.5, 15.1, 28.5, 21.4, 17.7, 6.42, 84.9])
    gamma_model = UVGamma(name='custom_gamma')
    params0 = [2.0, 0.1]
    fitted_params = gamma_model.fit(tstData, *params0, floc=0, fscale=1)
    print(fitted_params)
    # GMM fit
    gamma_model.setupGMM(tstData, nMoM=5)
    gmm_params = gamma_model.internalGMM.fit(params0, maxiter=4, optim_method='nm', wargs=dict(centered=False))
    print(gmm_params.summary())
    print("---- GMM params ----")
    print(gmm_params.params)

if __name__ == "__main__":
    example()
