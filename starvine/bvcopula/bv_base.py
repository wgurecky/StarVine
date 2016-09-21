##
# \brief Bivariate distribution base class.
from __future__ import print_function
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.stats import gaussian_kde
from scipy.stats.mstats import rankdata
# COPULA IMPORTS
from copula.t_copula import StudentTCopula
from copula.gauss_copula import GaussCopula
from copula.frank_copula import FrankCopula
from copula.gumbel_copula import GumbelCopula
from copula.clayton_copula import ClaytonCopula
from copula.indep_copula import IndepCopula


class BVbase(object):
    """!
    @brief Stores bivariate data.
    Contains methods to:
    - rank data transform
    - rotate data
    - remove nan or inf data points
    - plot bivarate data
    - fit copula to bivariate (ranked) data
    - compute basic bivariate statistics (eg. kendall's tau)

    Note: Depends on pandas for some useful statistical and plotting
    functionality.
    """
    def __init__(self, x, y, weights=None, **kwargs):
        """!
        @brief Bivariate data set init.
        @param x  <np_1darray> first marginal data set
        @param y  <np_1darray> second marginal data set
        @param weights <np_1darray> (optional) data weights
               normalized or unormalized weights accepted
        Note: len(u) == len(v) == len(weights)
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.u, self.v = None, None  # ranked data
        # normalize weights (weights must sum to 1.0)
        self.weights = np.array(weights)
        if self.weights:
            self.weights = self.weights / np.sum(self.weights)
        # init default copula family
        defaultFamily = ['t', 'gauss', 'frank', 'clayton', 'gumbel']
        self.setTrialCopula(kwargs.pop("family", defaultFamily))
        # Rank transform data
        self.rank(kwargs.pop("rankMethod", 0))
        # default rotation
        self.setRotation(kwargs.pop("rotation", 0))
        self.rotateData(self.u, self.v)

    def rank(self, method=0):
        """!
        @brief rank transfom the data
        @param method <int>
               if == 0: use standard rank transform,
               else: use CDF data transform.
        """
        self.rankMethod = method
        if method == 0:
            self.u = rankdata(self.x) / (len(self.x) + 1)
            self.v = rankdata(self.y) / (len(self.y) + 1)
        else:
            # use alternate CDF rank transform method
            kde_x = gaussian_kde(self.x)
            kde_y = gaussian_kde(self.y)
            u_hat = np.zeros(len(self.x))
            v_hat = np.zeros(len(self.y))
            for i, (xp, yp) in enumerate(zip(self.x, self.y)):
                u_hat[i] = kde_x.integrate_box_1d(-np.inf, xp)
                v_hat[i] = kde_y.integrate_box_1d(-np.inf, yp)
            self.u = u_hat
            self.v = v_hat

    def rankInv(self):
        """
        @brief Inverse rank transform data
        back to original scale.
        """
        pass

    def setTrialCopula(self, family=['t', 'gauss', 'clayton', 'gumbel', 'frank']):
        self.copulaBank = {'t': StudentTCopula(),
                           'gauss': GaussCopula(),
                           'indep': IndepCopula(),
                           'gumbel': GumbelCopula(),
                           'frank': FrankCopula(),
                           'clayton': ClaytonCopula(),
                           }
        self.trialFamily = family

    def empKTau(self):
        """!
        @brief Returns emperical kendall's tau of rank transformed data.
        @return <float> Kendall's tau rank correlation coeff
        """
        self.empKTau_, self.pval_ = kendalltau(self.u, self.v)
        return self.empKTau_, self.pval_

    def empSRho(self):
        """!
        @brief Returns emperical spearman rho, the rank correlation coefficient.
        @return <float> Spearman's rank correlation coeff
        """
        self.empSRho_, self.pval_ = spearmanr(self.u, self.v)
        return self.empSRho_, self.pval_

    def empPRho(self):
        """!
        @brief Returns linear correlation coefficient, pearson's rho.
        @return <float> pearson's correlation coefficient
        """
        self.empPRho_, self.pval_ = pearsonr(self.x, self.y)
        return self.empPRho_, self.pval_

    def copulaTournament(self, criterion='AIC'):
        """!
        @brief Determines the copula that best fits the rank transformed data
        based on the AIC criterion.
        All copula in the trial_family set are considered.
        """
        self.empKTau()
        if self.pval_ >= 0.99:
            print("Independence Coplua selected")
            return self.copulaBank['indep']
        # Find best fitting copula as judged by the AIC
        maxAIC, goldCopula, goldParams = 0, None, None
        for trialCopulaName in self.trialFamily:
            print("Fitting trial copula " + trialCopulaName + "...", end="")
            copula = self.copulaBank[trialCopulaName]
            fittedCopulaParams = self.fitCopula(copula)
            trialAIC = abs(fittedCopulaParams[2])
            print(" |AIC|: " + str(trialAIC))
            if trialAIC > maxAIC:
                goldCopula = copula
                goldParams = fittedCopulaParams
                maxAIC = trialAIC
        print(goldCopula.name + " copula selected")
        self.copulaModel = goldCopula
        self.copulaParams = goldParams
        return (self.copulaModel, self.copulaParams)

    def fitCopula(self, copula, thetaGuess=(None,None,)):
        """!
        @brief fit specified copula to data.
        @return (copula type <string>, fitted copula params <np_array>)
        """
        thetaHat = copula.fitMLE(self.UU, self.VV, 0, *thetaGuess)
        AIC = copula._AIC(self.UU, self.VV, 0, *thetaHat)
        return (copula.name, thetaHat, AIC, self.rotation)

    def rotateData(self, u, v, rotation=0):
        """!
        @brief Rotates the ranked data on the unit square.
        Allows for modeling of negative dependence with the
        standard archimedean copulas.
        @param u  Ranked data vector
        @param v  Ranked data vector
        @param rotation <int>
        """
        if rotation > 0:
            self.setRotation(rotation)
        self.UU = np.zeros(u.shape)  # storage for rotated u
        self.VV = np.zeros(v.shape)  # storage for rotated v
        if self.rotation == 1:
            # 90 degree rotation (flip U)
            self.UU = 1.0 - u
            self.VV = v + 1 - 1
        elif self.rotation == 2:
            # 180 degree rotation (flip U, flip V)
            self.UU = 1.0 - u
            self.VV = 1.0 - v
        elif self.rotation == 3:
            # 270 degree rotation (flip V)
            self.UU = u + 1 - 1
            self.VV = 1.0 - v
        else:
            self.UU = u + 1 - 1
            self.VV = v + 1 - 1
        return (self.UU, self.VV)

    def setRotation(self, rotation=0):
        """!
        @brief  Set the copula's orientation:
            0 == 0 deg
            1 == 90 deg rotation
            2 == 180 deg rotation
            3 == 270 deg rotation
        Allows for modeling negative dependence with the
        frank, gumbel, and clayton copulas (Archimedean Copula family is
        non-symmetric)
        """
        self.rotation = rotation
