##
# @brief  Pipeline interface for using with sklearn pipelines
##
from __future__ import print_function, absolute_import, division
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from starvine.vine.C_vine import Cvine


class VineModel(BaseEstimator):
    """!
    @brief  Implements the fit, fit_transform interface of the sklearn
    pipeline workflow for vine copula.
    """
    def __init__(self, vine_type='c',
                 trial_copula={},
                 copula_fit_method='mle',
                 vine_fit_method='ktau',
                 rank_transform=True):
        self.trial_copula_dict = trial_copula
        self._rank_transform = rank_transform
        if vine_type == 'c':
            self._vine = Cvine([])
        else:
            raise NotImplementedError

    def fit(self, X, weights=None):
        """
        @brief Fit vine copula model to data.
        @param X corrolated vectors with shape (Nsamples, Ndim). Can be
            unranked or ranked data
        """
        if not isinstance(X, pd.DataFrame):
            tstData = pd.DataFrame(X)
        else:
            tstData = X
        if self.rank_transform:
            x_r = tstData.dropna().rank()/(len(tstData)+1)
        else:
            x_r = tstData
        self.vine = Cvine(x_r, trial_copula=self.trial_copula_dict)
        self.vine.constructVine()

    def predict(self, n):
        """!
        @brief Predict correlated output vectors given uncorrolated inputs.
        @param n int. Number of samples to draw.
        """
        return self.vine.sample(n)

    @property
    def rank_transform(self):
        return self._rank_transform

    @rank_transform.setter
    def rank_transform(self, rt):
        self._rank_transform = bool(rt)

