##
# \brief Example plots of marshall-olkin and frank copula
# Also demonstrates how to draw scaled and unscaled samples from
# a copula model.
# Reproduces results from the paper:
# Using Copulas to model Dependence in Simulation Risk Assessment
# by Dana L. Kelly (Nov 2007), ASME International ME Congress and
# Exposition.
##
import numpy as np
# COPULA IMPORTS
import context
from starvine.bvcopula.copula import *
from starvine.uvar.uvmodels import *
from bvcopula import bv_plot

def olkin():
    rand_u = np.linspace(1e-9, 1-1e-9, 20)
    rand_v = np.linspace(1e-9, 1-1e-9, 20)

    u, v = np.meshgrid(rand_u, rand_v)

    # MARSHALL-OLKIN  model
    c = marshall_olkin_copula.OlkinCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[0.5, 0.75])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="olkin_copula_cdf.png")

    # draw and plot samples from copula
    u_cop, v_cop = c.sample(1000, *[0.5, 0.75])
    bv_plot.bvJointPlot(u_cop, v_cop, savefig="olkin_samples.png")

    # construct univariate model
    um = uv_exp.UVExp()
    um1 = um(2e-3)  # freeze univariate model
    um2 = um(2e-3)

    # draw scaled samples from copula
    u_cop_scaled, v_cop_scaled = c.sampleScale(um1, um2, 1000, *[0.5, 0.75])
    bv_plot.bvJointPlot(u_cop_scaled, v_cop_scaled, savefig="olkin_samples_scaled.png", s=5)


def frank():
    rand_u = np.linspace(1e-9, 1-1e-9, 20)
    rand_v = np.linspace(1e-9, 1-1e-9, 20)

    u, v = np.meshgrid(rand_u, rand_v)

    # MARSHALL-OLKIN  model
    c = frank_copula.FrankCopula()
    p = c.cdf(u.flatten(), v.flatten(), *[10.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, savefig="frank_copula_cdf.png")

    # draw and plot samples from copula
    u_cop, v_cop = c.sample(1000, *[10.])
    bv_plot.bvJointPlot(u_cop, v_cop, savefig="frank_samples.png")

    # construct univariate model
    um = uv_exp.UVExp()
    um1 = um(2e-3)  # freeze univariate model
    um2 = um(2e-3)

    # draw scaled samples from copula
    u_cop_scaled, v_cop_scaled = c.sampleScale(um1, um2, 1000, *[10.])
    bv_plot.bvJointPlot(u_cop_scaled, v_cop_scaled, savefig="frank_samples_scaled.png", s=5)


def plot_pdfs():
    rand_u = np.linspace(0.05, 0.95, 40)
    rand_v = np.linspace(0.05, 0.95, 40)

    u, v = np.meshgrid(rand_u, rand_v)

    # PLOT ALL CDF
    c = marshall_olkin_copula.OlkinCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[0.5, 0.75])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"olkin copula, $\theta$: %.2f, $DoF$: %.2f" % (0.50, 0.75), savefig="olkin_copula_pdf.png")

    c = frank_copula.FrankCopula()
    p = c.pdf(u.flatten(), v.flatten(), *[10.0])
    bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Frank, $\theta: %.2f$" % (10.0), savefig="frank_copula_pdf.png")


def main():
    olkin()
    frank()
    plot_pdfs()


if __name__ == "__main__":
    main()
