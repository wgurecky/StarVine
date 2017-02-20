#!/usr/env/python2
from __future__ import absolute_import, print_function, division
# starvine imports
import context
from starvine.uvar.uvmodel_factory import Uvm
from starvine.mvar import mvd
from starvine.bvcopula import PairCopula
from starvine.vine.C_vine import Cvine
from starvine.mvar.mv_plot import matrixPairPlot
import starvine.bvcopula as bvc
import matplotlib.pyplot as plt
#
from scipy.optimize import bisect, newton
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import tables as pt
#
# PyMAMBA
# from mamba import Mamba1d


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

    # SPAN
    tsat = -618.5
    zones = range(65, 98)
    for zone in zones:
        zBounds = bounds.read()[:, zone][~np.isnan(bounds.read()[:, zone])]
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
        upper_z, lower_z = zBounds
        bounds_label = str(lower_z) + "_" + str(upper_z)
        # span_1_mvd.plot(savefig=bounds_label + "_span.png", kde=False)

        # Construct Cvine
        lowerData = pd.DataFrame({"t": temps, "tke": tkes, "q": bhfs})
        lowerVine = Cvine(pd.DataFrame({"tke": tkes, "t": temps, "q": bhfs}))
        lowerVine.constructVine()

        # Sample Cvine
        lowerVineSamples = lowerVine.sample(n=500)
        matrixPairPlot(lowerVineSamples, savefig="singlePinPlots/" + bounds_label + "_vine_samples.png")
        ranked_data = lowerData.dropna().rank()/(len(lowerData)+1)
        # matrixPairPlot(ranked_data, savefig="singlePinPlots/" + bounds_label + "_ranked_samples.png")
        t_hat_vine, tke_hat_vine, q_hat_vine = lowerVineSamples['t'], lowerVineSamples['tke'], lowerVineSamples['q']

        kde_cdf = gaussian_kde(temps).integrate_box
        resampled_t = icdf_uv_bisect(temps, t_hat_vine, kde_cdf)
        kde_cdf = gaussian_kde(tkes).integrate_box
        resampled_tke = icdf_uv_bisect(tkes, tke_hat_vine, kde_cdf)
        # bvc.bvJointPlot(resampled_t, resampled_tke, vs=[temps, tkes],
        #                 savefig="singlePinPlots/" + bounds_label + "_t_tke_resampled.png")

        # Grow crud at resampled points
        #crudModel = Mamba1d(len(resampled_t))

        # Compare resampled crud to original crud result

    # Clean up
    store.close()


def t_tke():
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

    # SPAN
    tsat = -618.5
    zones = range(1, 98)
    results = []
    for zone in zones:
        zBounds = bounds.read()[:, zone][~np.isnan(bounds.read()[:, zone])]
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
        upper_z, lower_z = zBounds
        centroid_z = (upper_z + lower_z) / 2.
        bounds_label = str(lower_z) + "_" + str(upper_z)
        # span_1_mvd.plot(savefig=bounds_label + "_span.png", kde=False)

        # Load data into dataframe
        lowerData = pd.DataFrame({"t": temps, "tke": tkes})

        # Construct Copula
        my_copula = PairCopula(lowerData['t'], lowerData['tke'])
        copulaFamily = {'gauss': 0}
        my_copula.setTrialCopula(copulaFamily)

        # Obtain kendall's tau
        my_copula.copulaTournament()
        empRho, empPval = my_copula.empKTau()

        # Construct (guassian) margin models
        params0 = [2.0, 0.1]
        t_margin = Uvm("gauss")
        tv = t_margin.fitMLE(lowerData['t'], params0, bounds=((-5, 5), (1e-9, 10)))
        tke_margin = Uvm("gauss")
        tkev = tke_margin.fitMLE(lowerData['tke'], params0, bounds=((-5, 5), (1e-9, 10)))

        # obtain variance of each margin
        t_var = tv[1]
        tke_var = tkev[1]

        # append results
        print(centroid_z)
        results.append([centroid_z, empRho, t_var, tke_var])

        # Sample TH from copula

        # Grow crud at sampled TH conds
    results = np.array(results)
    np.savetxt("single_pin.txt", results)


def icdf_uv_bisect(ux, X, marginalCDFModel):
    """
    @brief Apply marginal model.
    """
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


if __name__ == "__main__":
    # main()
    t_tke()
