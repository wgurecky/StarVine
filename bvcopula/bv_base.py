##
# \brief Bivariate distribution base class.
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.stats import gaussian_kde
from scipy.stats.mstats import rankdata


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
        @param u  <np_1darray> first marginal data set
        @param v  <np_1darray> second marginal data set
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
        defaultFamily = ['t', 'gauss', 'indep', 'frank', 'clayton', 'gumbel']
        self.setTrialCopula(kwargs.pop("family", defaultFamily))
        # Rank transform data
        self.rank(kwargs.pop("rankMethod", 0))

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

    def setTrialCopula(self, family=['t', 'clayton', 'gauss', 'indep', 'frank']):
        self.trialFamily = family

    def empKTau(self):
        """!
        @brief Returns emperical kendall's tau of rank transformed data.
        @return <float> Kendall's tau rank correlation coeff
        """
        self.empKTau = kendalltau(self.u, self.v)
        return self.empKTau

    def empSRho(self):
        """!
        @brief Returns emperical spearman rho, the rank correlation coefficient.
        @return <float> Spearman's rank correlation coeff
        """
        self.empSRho = spearmanr(self.u, self.v)
        return self.empSRho

    def empPRho(self):
        """!
        @brief Returns linear correlation coefficient, pearson's rho.
        @return <float> pearson's correlation coefficient
        """
        self.empPRho = pearsonr(self.x, self.y)
        return self.empPRho

    def copulaTournament(self):
        """!
        @brief Determines the copula that best fits the rank transformed data
        All copula in the trial_family set are considered.
        """
        pass

    def rotateData(self, u, v, rotation=0):
        """!
        @brief Rotates the ranked data on the unit square.
        Allows for modeling of negative dependence with the
        standard archemidean copulas.
        @param u  Ranked data vector
        @param v  Ranked data vector
        @param rotation <int>
        """
        self.rotation = rotation
        self.UU = np.zeros(u.shape)  # storage for rotated u
        self.VV = np.zeros(v.shape)  # storage for rotated v
        if rotation == 1:
            # 90 degree rotation
            self.UU = 1.0 - u
            self.VV = v
        elif rotation == 2:
            # 180 degree rotation
            self.UU = 1.0 - u
            self.VV = 1.0 - v
        elif rotation == 3:
            # 270 degree rotation
            self.UU = u
            self.VV = 1.0 - v
        else:
            self.UU = u
            self.VV = v
        return (self.UU, self.VV)

    def _invRotateData(self):
        """!
        @brief Rotates data back it's original orientation.
        """
        if self.rotation == 1:
            # -90 degree rotation
            self.UU = self.UU - 1.0
            self.VV = self.VV
        elif self.rotation == 2:
            # -180 degree rotation
            self.UU = self.UU - 1.0
            self.VV = self.VV - 1.0
        elif self.rotation == 3:
            # -270 degree rotation
            self.UU = self.UU
            self.VV = self.VV - 1.0
        else:
            pass
        self.rotation = 0
        return (self.UU, self.VV)
