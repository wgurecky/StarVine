#!/usr/env/python2
from __future__ import absolute_import, print_function, division
# starvine imports
import context
from starvine.mvar import mvd
from starvine.vine.C_vine import Cvine
from starvine.mvar.mv_plot import matrixPairPlot
import starvine.bvcopula as bvc
#
from scipy.optimize import bisect, newton
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
# copula imports
import tables as pt


def h5Load(store, grpName):
    """!
    @brief Load dataset from H5 file.
    @param store <pd.HDFStore> instance
    @param grpName HDF5 group name locating data in h5 store
    """
    if grpName[0] is not "/":
        grpName = "/" + grpName
    data = store.get_node(grpName, "block0_values")
    return data


def main():
    # read data from external h5 file
    h5file = 'Cicada_cfd_180x_cht.h5.post.binned.h5'
    # store = pd.HDFStore(h5file)
    store = pt.open_file(h5file)
    bounds = h5Load(store, "Water/UO2 [Interface 1]/Temperature_bounds")
    temperature = h5Load(store, "Water/UO2 [Interface 1]/Temperature")
    tke = h5Load(store, "Water/UO2 [Interface 1]/TurbulentKineticEnergy")
    crud_thick = h5Load(store, "Water/UO2 [Interface 1]/CrudThickness")
    b10 = h5Load(store, "Water/UO2 [Interface 1]/CrudBoronDensity")
    weight = h5Load(store, "Water/UO2 [Interface 1]/Temperature_weights")
    bhf = h5Load(store, "Water/UO2 [Interface 1]/BoundaryHeatFlux")

    """
    # create multi-variate dataset for span 1
    # for zone in range(69, 81):
    for zone in range(69, 78):
        lower_b = bounds.read()[:, zone][0]
        print("Generating plot for zone: " + str(zone))
        temps = temperature.read()[:, zone][~np.isnan(temperature.read()[:, zone])]
        tkes = tke.read()[:, zone][~np.isnan(tke.read()[:, zone])]
        cruds = crud_thick.read()[:, zone][~np.isnan(crud_thick.read()[:, zone])]
        b10s = b10.read()[:, zone][~np.isnan(b10.read()[:, zone])]
        bhfs = bhf.read()[:, zone][~np.isnan(bhf.read()[:, zone])]
        weights = weight.read()[:, zone][~np.isnan(weight.read()[:, zone])]
        span_1_dataDict = {"Residual Temperature [K]": temps,
                           "Residual TKE [J/kg]": tkes,
                           "Residual BHF [W/m^2]": bhfs,
                           }
        span_1_mvd = mvd.Mvd()
        span_1_mvd.setData(span_1_dataDict, weights)
        span_1_mvd.plot(savefig="mvd_" + str(round(lower_b, 3)) + ".png", kde=False)
    """

    # upper span plot
    tsat = -618.5
    zones = range(72, 74)
    temps = temperature.read()[:, zones][~np.isnan(temperature.read()[:, zones])]
    tkes = tke.read()[:, zones][~np.isnan(tke.read()[:, zones])]
    cruds = crud_thick.read()[:, zones][~np.isnan(crud_thick.read()[:, zones])]
    b10s = b10.read()[:, zones][~np.isnan(b10.read()[:, zones])]
    bhfs = bhf.read()[:, zones][~np.isnan(bhf.read()[:, zones])]
    weights = weight.read()[:, zones][~np.isnan(weight.read()[:, zones])]
    span_1_dataDict = {"Residual Temperature [K]": temps,
                       "Residual TKE [J/kg]": tkes,
                       "Residual BHF [W/m^2]": bhfs,
                       }
    span_1_mvd = mvd.Mvd()
    span_1_mvd.setData(span_1_dataDict, weights)
    span_1_mvd.plot(savefig="upper_span.png", kde=False)

    # fit bivariate copula to span plot; T vs TKE:
    # copula = bvc.PairCopula(temps, tkes)
    # copula.copulaTournament()

    # init Cvine
    print("================= Construct Upper Vine =================")
    upperData = pd.DataFrame({"t": temps, "tke": tkes, "q": bhfs})
    upperVine = Cvine(pd.DataFrame({"t": temps, "tke": tkes, "q": bhfs}))
    upperVine.constructVine()
    upperVine.plotVine(savefig="upper_vine.png")
    print("========================================================")
    upperVineSamples = upperVine.sample(n=500)
    matrixPairPlot(upperVineSamples, savefig="upper_vine_samples.png")
    upper_ranked_data = upperData.dropna().rank()/(len(upperData)+1)
    matrixPairPlot(upper_ranked_data, savefig="upper_ranked_samples.png")
    t_hat_vine, tke_hat_vine, q_hat_vine = upperVineSamples['t'], upperVineSamples['tke'], upperVineSamples['q']

    # plot original
    # bvc.bvJointPlot(temps, tkes, savefig="upper_t_tke_original.png")

    # sample from copula
    # print("Copula Params: " + str(copula.copulaParams))
    # t_hat, tke_hat = copula.copulaModel.sample(500)
    # bvc.bvJointPlot(t_hat_vine, tke_hat_vine, savefig="upper_t_tke_copula_sample.png")

    # rand_u = np.linspace(0.05, 0.95, 40)
    # rand_v = np.linspace(0.05, 0.95, 40)
    # u, v = np.meshgrid(rand_u, rand_v)
    # copula_pdf = copula.copulaModel.pdf(u.flatten(), v.flatten())
    # bvc.bvContourf(u.flatten(), v.flatten(), copula_pdf, savefig="upper_t_tke_copula_pdf.png")

    # Resample original data
    def icdf_uv_bisect(ux, X, marginalCDFModel):
        icdf = np.zeros(np.array(X).size)
        for i, xx in enumerate(X):
            kde_cdf_err = lambda m: xx - marginalCDFModel(-np.inf, m)
            try:
                icdf[i] = bisect(kde_cdf_err,
                                 min(ux) - np.abs(0.5 * min(ux)),
                                 max(ux) + np.abs(0.5 * max(ux)),
                                 xtol=1e-3, maxiter=15)
                icdf[i] = newton(kde_cdf_err, icdf[i], tol=1e-6, maxiter=20)
            except:
                icdf[i] = np.nan
        return icdf
    kde_cdf = gaussian_kde(temps).integrate_box
    resampled_t = icdf_uv_bisect(temps, t_hat_vine, kde_cdf)
    kde_cdf = gaussian_kde(tkes).integrate_box
    resampled_tke = icdf_uv_bisect(tkes, tke_hat_vine, kde_cdf)
    bvc.bvJointPlot(resampled_t, resampled_tke, vs=[temps, tkes], savefig="upper_t_tke_resampled.png")

    # LOWER SPAN
    tsat = -618.5
    zones = range(70, 72)
    temps = temperature.read()[:, zones][~np.isnan(temperature.read()[:, zones])]
    tkes = tke.read()[:, zones][~np.isnan(tke.read()[:, zones])]
    cruds = crud_thick.read()[:, zones][~np.isnan(crud_thick.read()[:, zones])]
    b10s = b10.read()[:, zones][~np.isnan(b10.read()[:, zones])]
    bhfs = bhf.read()[:, zones][~np.isnan(bhf.read()[:, zones])]
    weights = weight.read()[:, zones][~np.isnan(weight.read()[:, zones])]
    span_1_dataDict = {"Residual Temperature [K]": temps,
                       "Residual TKE [J/kg]": tkes,
                       "Residual BHF [W/m^2]": bhfs,
                       }
    span_1_mvd = mvd.Mvd()
    span_1_mvd.setData(span_1_dataDict, weights)
    span_1_mvd.plot(savefig="lower_span.png", kde=False)

    # fit bivariate copula to span plot; T vs TKE:
    # copula = bvc.PairCopula(temps, tkes)
    # copula.copulaTournament()

    # init Cvine
    print("================= Construct Lower Vine =================")
    lowerData = pd.DataFrame({"t": temps, "tke": tkes, "q": bhfs})
    lowerVine = Cvine(pd.DataFrame({"t": temps, "tke": tkes, "q": bhfs}))
    lowerVine.constructVine()
    lowerVine.plotVine(savefig="lower_vine.png")
    print("========================================================")
    lowerVineSamples = lowerVine.sample(n=500)
    matrixPairPlot(lowerVineSamples, savefig="lower_vine_samples.png")
    lower_ranked_data = lowerData.dropna().rank()/(len(lowerData)+1)
    matrixPairPlot(lower_ranked_data, savefig="lower_ranked_samples.png")
    t_hat_vine, tke_hat_vine, q_hat_vine = lowerVineSamples['t'], lowerVineSamples['tke'], lowerVineSamples['q']

    # plot original
    # bvc.bvJointPlot(temps, tkes, savefig="lower_t_tke_original.png")

    # sample from copula
    # print("Copula Params: " + str(copula.copulaParams))
    # t_hat, tke_hat = copula.copulaModel.sample(500)
    # bvc.bvJointPlot(t_hat_vine, tke_hat_vine, savefig="lower_t_tke_copula_sample.png")

    # rand_u = np.linspace(0.05, 0.95, 40)
    # rand_v = np.linspace(0.05, 0.95, 40)
    # u, v = np.meshgrid(rand_u, rand_v)
    # copula_pdf = copula.copulaModel.pdf(u.flatten(), v.flatten())
    # bvc.bvContourf(u.flatten(), v.flatten(), copula_pdf, savefig="lower_t_tke_copula_pdf.png")

    # Resample original data
    def icdf_uv_bisect(ux, X, marginalCDFModel):
        icdf = np.zeros(np.array(X).size)
        for i, xx in enumerate(X):
            kde_cdf_err = lambda m: xx - marginalCDFModel(-np.inf, m)
            try:
                icdf[i] = bisect(kde_cdf_err,
                                 min(ux) - np.abs(0.5 * min(ux)),
                                 max(ux) + np.abs(0.5 * max(ux)),
                                 xtol=1e-2, maxiter=10)
                icdf[i] = newton(kde_cdf_err, icdf[i], tol=1e-6, maxiter=20)
            except:
                icdf[i] = np.nan
        return icdf
    kde_cdf = gaussian_kde(temps).integrate_box
    resampled_t = icdf_uv_bisect(temps, t_hat_vine, kde_cdf)
    kde_cdf = gaussian_kde(tkes).integrate_box
    resampled_tke = icdf_uv_bisect(tkes, tke_hat_vine, kde_cdf)
    bvc.bvJointPlot(resampled_t, resampled_tke, vs=[temps, tkes], savefig="lower_t_tke_resampled.png")

    # Clean up
    store.close()


if __name__ == "__main__":
    main()


