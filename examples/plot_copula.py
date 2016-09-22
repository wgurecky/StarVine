##
# \brief Example plots of all implemented copula
import numpy as np
# COPULA IMPORTS
import context
from starvine.bvcopula.copula import *
from bvcopula import bv_plot
# Plotting

def plot_cdfs():
    rand_u = np.linspace(1e-9, 1-1e-9, 20)
    rand_v = np.linspace(1e-9, 1-1e-9, 20)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    c = t_copula.StudentTCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[0.7, 10])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="t_copula_cdf.png")

    c = gauss_copula.GaussCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[0.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gauss_copula_cdf.png")

    c = frank_copula.FrankCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="frank_copula_cdf.png")

    c = clayton_copula.ClaytonCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="clayton_copula_cdf.png")

    c = gumbel_copula.GumbelCopula()
    p = c.cdf(u.flatten(), v.flatten(), 0, *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gumbel_copula_cdf.png")


def plot_pdfs():
    rand_u = np.linspace(0.05, 0.95, 40)
    rand_v = np.linspace(0.05, 0.95, 40)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    c = t_copula.StudentTCopula()
    p = c.pdf(u.flatten(), v.flatten(), 0, [0.2, 10])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="t_copula_pdf.png")

    c = gauss_copula.GaussCopula()
    p = c.pdf(u.flatten(), v.flatten(), 0, [0.2])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gauss_copula_pdf.png")

    c = frank_copula.FrankCopula()
    p = c.pdf(u.flatten(), v.flatten(), 0, [9.2])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="frank_copula_pdf.png")

    c = clayton_copula.ClaytonCopula()
    p = c.pdf(u.flatten(), v.flatten(), 0, [1.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="clayton_copula_pdf.png")

    c = gumbel_copula.GumbelCopula()
    p = c.pdf(u.flatten(), v.flatten(), 0, [2.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gumbel_copula_pdf.png")


def main():
    plot_cdfs()
    plot_pdfs()


if __name__ == "__main__":
    main()
