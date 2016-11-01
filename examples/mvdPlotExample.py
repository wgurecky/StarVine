#!/usr/env/python2
from __future__ import absolute_import, print_function, division
import context
from starvine.mvar import mvd
import pandas as pd
import numpy as np


def h5Load(store, grpName):
    """!
    @brief Load dataset from H5 file.
    @param store <pd.HDFStore> instance
    @param grpName HDF5 group name locating data in h5 store
    """
    data = store[grpName]
    metadata = store.get_storer(grpName).attrs.metadata
    return data, metadata

def main():
    # read data from external h5 file
    h5file = 'Cicada_cfd_180x_cht.h5.post.binned.h5'
    store = pd.HDFStore(h5file)
    bounds = h5Load(store, "Water/UO2 [Interface 1]/Temperature_bounds")[0]
    temperature = h5Load(store, "Water/UO2 [Interface 1]/Temperature")[0]
    tke = h5Load(store, "Water/UO2 [Interface 1]/TurbulentKineticEnergy")[0]
    crud_thick = h5Load(store, "Water/UO2 [Interface 1]/CrudThickness")[0]
    b10 = h5Load(store, "Water/UO2 [Interface 1]/CrudBoronDensity")[0]
    weight = h5Load(store, "Water/UO2 [Interface 1]/Temperature_weights")[0]
    bhf = h5Load(store, "Water/UO2 [Interface 1]/BoundaryHeatFlux")[0]
    store.close()

    # create multi-variate dataset for span 1
    # for zone in range(68, 80):
    for zone in range(69, 80):
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

    # full span plot
    tsat = -618.5
    zones = range(71, 78)
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
    span_1_mvd.setData(span_1_dataDict, weights)
    span_1_mvd.plot(savefig="mvd_span.png", kde=False)

    """
    # create multi-variate dataset for span 2
    span_2_dataDict = {"bounds": bounds[0].values[:,2],
                       "temperature": temperature[0].values[:,2],
                       "tke": tke[0].values[:,2],
                       "crud": crud_thick[0].values[:,2],
                       "b10": b10[0].values[:,2],
                       }

    # create multi-variate dataset for span 3
    span_3_dataDict = {"bounds": bounds[0].values[:,3],
                       "temperature": temperature[0].values[:,3],
                       "tke": tke[0].values[:,3],
                       "crud": crud_thick[0].values[:,3],
                       "b10": b10[0].values[:,3],
                       }
    """


if __name__ == "__main__":
    main()


