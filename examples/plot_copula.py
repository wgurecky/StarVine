##
# \brief Example plots of all implemented copula
import numpy as np
# COPULA IMPORTS
import context
from starvine.bvcopula.copula import *
from bvcopula import bv_plot
# from copula.gauss_copula import GaussCopula
# from copula.frank_copula import FrankCopula
# from copula.gumbel_copula import GumbelCopula
# from copula.clayton_copula import ClaytonCopula
# from copula.indep_copula import IndepCopula
# Plotting


def main():
    """!
    @brief Plot all copula PDF and CDFs
    """
    rand_u = np.linspace(1e-9, 1-1e-9, 20)
    rand_v = np.linspace(1e-9, 1-1e-9, 20)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    c = t_copula.StudentTCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[0.7, 10])
    bv_plot.bvContour(u.flatten(), v.flatten(), p, savefig="t_copula_cdf.png")

    """
    c = gauss_copula.GaussCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[0.7])
    bv_plot.bvContour(u.flatten(), v.flatten(), p, savefig="gauss_copula_cdf.png")

    c = frank_copula.FrankCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContour(u.flatten(), v.flatten(), p, savefig="frank_copula_cdf.png")

    c = clayton_copula.ClaytonCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContour(u.flatten(), v.flatten(), p, savefig="clayton_copula_cdf.png")

    c = gumbel_copula.GumbelCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContour(u.flatten(), v.flatten(), p, savefig="gumbel_copula_cdf.png")
    """


if __name__ == "__main__":
    main()
