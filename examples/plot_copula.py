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
    p = c.cdf(u.flatten(), v.flatten(), *[0.7, 10])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="t_copula_cdf.png")

    c = gauss_copula.GaussCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[0.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gauss_copula_cdf.png")

    c = frank_copula.FrankCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="frank_copula_cdf.png")

    c = gumbel_copula.GumbelCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="gumbel_copula_cdf.png")

    # CLAYTON CDFS
    c = clayton_copula.ClaytonCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="clayton_copula_cdf.png")

    c_90 = clayton_copula.ClaytonCopula(1)
    p_90 = c_90.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_90, savefig="clayton_90_copula_cdf.png")

    c_180 = clayton_copula.ClaytonCopula(2)
    p_180 = c_180.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_180, savefig="clayton_180_copula_cdf.png")

    c_270 = clayton_copula.ClaytonCopula(3)
    p_270 = c_270.cdf(u.flatten(), v.flatten(), *[2.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_270, savefig="clayton_270_copula_cdf.png")



def plot_pdfs():
    rand_u = np.linspace(0.05, 0.95, 40)
    rand_v = np.linspace(0.05, 0.95, 40)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    c = t_copula.StudentTCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[0.01, 10.2])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"t copula, $\theta$: %.2f, $DoF$: %.1f" % (0.01, 10.0), savefig="t_copula_pdf.png")

    c = gauss_copula.GaussCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[-0.7])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Gauss, $\theta: %.2f$" % (-0.7), savefig="gauss_copula_pdf.png")

    # FRANK PDFS
    c = frank_copula.FrankCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[9.2])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Frank, $\theta: %.2f$" % (9.2), savefig="frank_copula_pdf.png")

    # GUMBEL PDFS
    c = gumbel_copula.GumbelCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[2.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Gumbel-$0^{\circ}$, $\theta: %.2f$" % (2.0), savefig="gumbel_copula_pdf.png")

    c = gumbel_copula.GumbelCopula(1)
    p = c.pdf(u.flatten(), v.flatten(), *[2.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Gumbel-$90^{\circ}$, $\theta: %.2f$" % (2.0), savefig="gumbel_90_copula_pdf.png")

    c = gumbel_copula.GumbelCopula(2)
    p = c.pdf(u.flatten(), v.flatten(), *[2.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Gumbel-$180^{\circ}$, $\theta: %.2f$" % (2.0), savefig="gumbel_180_copula_pdf.png")

    c = gumbel_copula.GumbelCopula(3)
    p = c.pdf(u.flatten(), v.flatten(), *[2.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Gumbel-$270^{\circ}$, $\theta: %.2f$" % (2.0), savefig="gumbel_270_copula_pdf.png")

    # CLAYTON PDFS
    c = clayton_copula.ClaytonCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[1.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Clayton-$0^{\circ}$, $\theta: %.2f$" % (1.0), savefig="clayton_copula_pdf.png")

    c_90 = clayton_copula.ClaytonCopula(1)
    p_90 = c_90.pdf(u.flatten(), v.flatten(), *[1.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_90, title=r"Clayton-$90^{\circ}$, $\theta: %.2f$" % (1.0), savefig="clayton_90_copula_pdf.png")

    c_180 = clayton_copula.ClaytonCopula(2)
    p_180 = c_180.pdf(u.flatten(), v.flatten(), *[1.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_180, title=r"Clayton-$180^{\circ}$, $\theta: %.2f$" % (1.0), savefig="clayton_180_copula_pdf.png")

    c_270 = clayton_copula.ClaytonCopula(3)
    p_270 = c_270.pdf(u.flatten(), v.flatten(), *[1.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p_270, title=r"Clayton-$270^{\circ}$, $\theta: %.2f$" % (1.0), savefig="clayton_270_copula_pdf.png")


def main():
    plot_cdfs()
    plot_pdfs()


if __name__ == "__main__":
    main()
