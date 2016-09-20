##
# \brief Example plots of all implemented copula
import numpy as np
# COPULA IMPORTS
import t_copula as tc
# from copula.gauss_copula import GaussCopula
# from copula.frank_copula import FrankCopula
# from copula.gumbel_copula import GumbelCopula
# from copula.clayton_copula import ClaytonCopula
# from copula.indep_copula import IndepCopula
# Plotting
from bv_plot import bvContour


def main():
    """!
    @brief Plot all copula PDF and CDFs
    """
    rand_u = np.random.uniform(1e-9, 1-1e-9, 10)
    rand_v = np.random.uniform(1e-9, 1-1e-9, 10)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    t_copula = tc.StudentTCopula()
    p = t_copula.cdf(u.flatten(), v.flatten(), 0, *[0.7, 10])
    bvContour(u.flatten(), v.flatten(), p, savefig="t_copula_cdf.png")
    #bvJointPlot(u, v, savefig="gauss_copula_cdf.png")
    #bvJointPlot(u, v, savefig="frank_copula_cdf.png")
    #bvJointPlot(u, v, savefig="clayton_copula_cdf.png")
    #bvJointPlot(u, v, savefig="gumbel_copula_cdf.png")

    # PLOT ALL PDF

if __name__ == "__main__":
    main()
