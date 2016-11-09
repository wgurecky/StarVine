#!/usr/env/python2
from __future__ import absolute_import, print_function, division
from scipy.optimize import bisect, newton
from scipy.stats import gaussian_kde
import context
from starvine.mvar import mvd
import pandas as pd
import numpy as np
# copula imports
import starvine.bvcopula as bvc


def h5Load(store, grpName):
    """!
    @brief Load dataset from H5 file.
    @param store <pd.HDFStore> instance
    @param grpName HDF5 group name locating data in h5 store
    """
    data = store[grpName]
    return data


def main():
    # read data from external h5 file
    h5file = 'Cicada_cfd_180x_cht.h5.post.binned.h5'
    store = pd.HDFStore(h5file)
    bounds = h5Load(store, "Water/UO2 [Interface 1]/Temperature_bounds")
    temperature = h5Load(store, "Water/UO2 [Interface 1]/Temperature")
    tke = h5Load(store, "Water/UO2 [Interface 1]/TurbulentKineticEnergy")
    crud_thick = h5Load(store, "Water/UO2 [Interface 1]/CrudThickness")
    b10 = h5Load(store, "Water/UO2 [Interface 1]/CrudBoronDensity")
    weight = h5Load(store, "Water/UO2 [Interface 1]/Temperature_weights")
    bhf = h5Load(store, "Water/UO2 [Interface 1]/BoundaryHeatFlux")
    store.close()

    """
    # create multi-variate dataset for span 1
    # for zone in range(68, 80):
    for zone in range(71, 78):
        lower_b = bounds.values[:, zone][0]
        print("Generating plot for zone: " + str(zone))
        temps = temperature.values[:, zone][~np.isnan(temperature.values[:, zone])]
        tkes = tke.values[:, zone][~np.isnan(tke.values[:, zone])]
        cruds = crud_thick.values[:, zone][~np.isnan(crud_thick.values[:, zone])]
        b10s = b10.values[:, zone][~np.isnan(b10.values[:, zone])]
        bhfs = bhf.values[:, zone][~np.isnan(bhf.values[:, zone])]
        weights = weight.values[:, zone][~np.isnan(weight.values[:, zone])]
        span_1_dataDict = {"Residual Temperature [K]": temps,
                           "Residual TKE [J/kg]": tkes,
                           "Residual BHF [W/m^2]": bhfs,
                           }
        span_1_mvd = mvd.Mvd()
        span_1_mvd.setData(span_1_dataDict, weights)
        span_1_mvd.plot(savefig="mvd_" + str(round(lower_b, 3)) + ".png", kde=False)
    """

    # full span plot
    tsat = -618.5
    zones = range(71, 72)
    temps = temperature.values[:, zones][~np.isnan(temperature.values[:, zones])]
    tkes = tke.values[:, zones][~np.isnan(tke.values[:, zones])]
    cruds = crud_thick.values[:, zones][~np.isnan(crud_thick.values[:, zones])]
    b10s = b10.values[:, zones][~np.isnan(b10.values[:, zones])]
    bhfs = bhf.values[:, zones][~np.isnan(bhf.values[:, zones])]
    weights = weight.values[:, zones][~np.isnan(weight.values[:, zones])]
    span_1_dataDict = {"Residual Temperature [K]": temps,
                       "Residual TKE [J/kg]": tkes,
                       "Residual BHF [W/m^2]": bhfs,
                       }
    span_1_mvd = mvd.Mvd()
    #span_1_mvd.setData(span_1_dataDict, weights)
    #span_1_mvd.plot(savefig="mvd_span.png", kde=False)

    # fit bivariate copula to span plot; T vs TKE:
    copula = bvc.PairCopula(temps, tkes, family={"t": 0})
    copula.copulaTournament()

    # plot original
    bvc.bvJointPlot(temps, tkes, savefig="t_tke_original.png")

    # sample from copula
    print("Copula Params: " + str(copula.copulaParams))
    t_hat, tke_hat = copula.copulaModel.sample(500)
    bvc.bvJointPlot(t_hat, tke_hat, savefig="t_tke_copula_sample.png")

    rand_u = np.linspace(0.05, 0.95, 40)
    rand_v = np.linspace(0.05, 0.95, 40)
    u, v = np.meshgrid(rand_u, rand_v)
    copula_pdf = copula.copulaModel.pdf(u.flatten(), v.flatten())
    bvc.bvContourf(u.flatten(), v.flatten(), copula_pdf, savefig="t_tke_copula_pdf.png")

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
    resampled_t = icdf_uv_bisect(temps, t_hat, kde_cdf)
    kde_cdf = gaussian_kde(tkes).integrate_box
    resampled_tke = icdf_uv_bisect(tkes, tke_hat, kde_cdf)
    bvc.bvJointPlot(resampled_t, resampled_tke, savefig="t_tke_resampled.png")


if __name__ == "__main__":
    main()


