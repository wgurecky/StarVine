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
    h5file = 'Cicada_binned_ex.h5'
    store = pd.HDFStore(h5file)
    bounds = h5Load(store, "Water/UO2/Temperature_bounds")[0]
    temperature = h5Load(store, "Water/UO2/Temperature")[0]
    tke = h5Load(store, "Water/UO2/TurbulentKineticEnergy")[0]
    crud_thick = h5Load(store, "Water/UO2/CrudThickness")[0]
    b10 = h5Load(store, "Water/UO2/CrudBoronDensity")[0]
    store.close()

    # create multi-variate dataset for span 1
    # span_1_dataDict = {"bounds": bounds.values[:,1],
    # for zone in range(68, 80):
    for zone in range(69, 80):
        lower_b = bounds.values[:, zone][0]
        print("Generating plot for zone: " + str(zone))
        temps = temperature.values[:, zone][~np.isnan(temperature.values[:, zone])]
        tkes = tke.values[:, zone][~np.isnan(tke.values[:, zone])]
        cruds = crud_thick.values[:, zone][~np.isnan(crud_thick.values[:, zone])]
        b10s = b10.values[:, zone][~np.isnan(b10.values[:, zone])]
        editIdx = (cruds >= 1e-9)
        span_1_dataDict = {"Temperature [K]": temps[editIdx],
                           "TKE [J/kg]": tkes[editIdx],
                           "CRUD Thickness [micron]": cruds[editIdx] * 1e6,
                           "B10 Mass [g/cm^2]": b10s[editIdx] * 1e3 * 1e2,
                           }
        span_1_mvd = mvd.Mvd()
        span_1_mvd.setData(span_1_dataDict)
        span_1_mvd.plot(savefig="mvd_" + str(round(lower_b, 3)) + ".png")


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


