##
# \brief Pair Copula.
# Bivariate distribution base class.
from __future__ import print_function, absolute_import
import numpy as np
from six import iteritems
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.stats import gaussian_kde
from scipy.stats.mstats import rankdata
# COPULA IMPORTS
try:
    from starvine.bvcopula.copula_factory import Copula
except:
    from copula_factory import Copula


class PairCopula(object):
    """!
    @brief Stores bivariate data for pair copula construction.
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
        @param x  <b>np_1darray</b> first marginal data set
        @param y  <b>np_1darray</b> second marginal data set
        @param weights <b>np_1darray</b> (optional) data weights
               normalized or unormalized weights accepted
        Note: len(u) == len(v) == len(weights)
        """
        self.copulaModel, self.copulaParams = None, (None, None, )
        self.id = kwargs.pop("id", None)
        self.x = np.array(x)
        self.y = np.array(y)
        # normalize weights (weights must sum to 1.0)
        self.weights = weights
        if self.weights is not None:
            self.weights = self.weights / np.sum(self.weights)
        self.setTrialCopula(kwargs.pop("family", self.defaultFamily))
        # default data ranking method
        self.rank_method = kwargs.pop("rankMethod", 0)
        # default rotation
        self.setRotation(kwargs.pop("rotation", 0))

    def rank(self, method=0):
        """!
        @brief Compute ranks of the data
        @param method <b>int</b>
               if == 0: use standard rank transform,
               else: use CDF data transform.
        @return (u, v) tuple of <b>np_1darray</b> ranked samples
        """
        if method == 0:
            u = rankdata(self.x) / (len(self.x) + 1)
            v = rankdata(self.y) / (len(self.y) + 1)
        else:
            # use alternate CDF rank transform method
            kde_x = gaussian_kde(self.x)
            kde_y = gaussian_kde(self.y)
            u_hat = np.zeros(len(self.x))
            v_hat = np.zeros(len(self.y))
            for i, (xp, yp) in enumerate(zip(self.x, self.y)):
                u_hat[i] = kde_x.integrate_box_1d(-np.inf, xp)
                v_hat[i] = kde_y.integrate_box_1d(-np.inf, yp)
            u = u_hat
            v = v_hat
        return u, v

    @property
    def rank_method(self):
        return self._rank_method

    @rank_method.setter
    def rank_method(self, method):
        assert type(method) is int
        self._rank_method = method

    @property
    def u(self):
        """!
        @brief Ranked x samples
        @return <b>np_1darray</b>
        """
        return self.rank(self.rank_method)[0]

    @property
    def v(self):
        """!
        @brief Ranked y samples
        @return <b>np_1darray</b>
        """
        return self.rank(self.rank_method)[1]

    def rankInv(self):
        """!
        @brief Inverse rank transform data
        back to original scale.
        """
        pass

    def setTrialCopula(self, family):
        self.trialFamily = family
        self.copulaBank = {}
        for name, rotation in iteritems(self.trialFamily):
            self.copulaBank[name] = Copula(name, rotation)

    def empKTau(self):
        """!
        @brief Returns emperical kendall's tau of rank transformed data.
        @return <b>float</b> Kendall's tau rank correlation coeff
        """
        self.empKTau_, self.pval_ = kendalltau(self.UU, self.VV)
        return self.empKTau_, self.pval_

    def empSRho(self):
        """!
        @brief Returns emperical spearman rho, the rank correlation coefficient.
        @return <b>float</b> Spearman's rank correlation coeff
        """
        self.empSRho_, self.pval_ = spearmanr(self.u, self.v)
        return self.empSRho_, self.pval_

    def empPRho(self):
        """!
        @brief Returns linear correlation coefficient, pearson's rho.
        @return <b>float</b> pearson's correlation coefficient
        """
        self.empPRho_, self.pval_ = pearsonr(self.x, self.y)
        return self.empPRho_, self.pval_

    @staticmethod
    def empKc(UU, VV):
        """!
        @brief Compute empirical kendall's function.
        """
        z = np.zeros(len(UU))
        for i in range(len(UU)):
            result = 0.
            for j in range(len(VV)):
                if i == j:
                    continue
                if (UU[j] < UU[i]) & (VV[j] < VV[i]):
                    result += 1.
            z[i] = (1. / (len(UU) - 1.)) * result
        x2 = np.sort(z)
        f2 = np.array(range(len(z))) / float(len(z))
        return x2, f2

    def copulaTournament(self, criterion='AIC', **kwargs):
        """!
        @brief Determines the copula that best fits the rank transformed data
        based on the AIC or Kendall's function criterion.
        All Copula in self.trialFamily set are considered.
        """
        if criterion == 'Kc':
            t_emp, kc_emp = self.empKc(self.UU, self.VV)
        vb = kwargs.pop("verbosity", True)
        self.empKTau()
        if self.pval_ >= 0.05 and self.weights is None:
            print("Independence Coplua selected")
            goldCopula = self.copulaBank["gauss"]
            goldParams = self.fitCopula(goldCopula)
            self.copulaModel = goldCopula
            self.copulaParams = goldParams
            if vb: print("ID: %s. %s copula selected.  fitted params="
                         % (str(self.id), goldCopula.name) + str(goldParams[1]))
            if vb: print("-------------------------------------------")
            # return self.copulaBank['indep']
            return (self.copulaModel, self.copulaParams)
        # Find best fitting copula
        best_AIC, best_kc, goldCopula, goldParams = np.inf, np.inf, None, None
        for trialCopulaName, rotation in iteritems(self.trialFamily):
            if vb: print("Copula " + (trialCopulaName).ljust(12) + '.', end="")
            copula = self.copulaBank[trialCopulaName]
            fittedCopulaParams = self.fitCopula(copula)
            trialAIC = fittedCopulaParams[2]
            trial_kc_metric = 0
            if criterion == 'Kc':
                trial_kc_metric = self.compute_kc_metric(copula, t_emp, kc_emp)
                print(" KC_m: " + '{:+05.5f}'.format(trial_kc_metric), end=",")
            if vb: print(" AIC: " + '{:+05.3f}'.format(trialAIC), end=",")
            if vb: print(" emp_ktau: " + '{:+05.3f}'.format(self.empKTau_), end=",")
            if vb: print(" cop_ktau: " + \
                    '{:+05.3f}'.format(copula.kTau(copula.rotation, *fittedCopulaParams[1])))
            if trialAIC < best_AIC and criterion == 'AIC':
                goldCopula = copula
                goldParams = fittedCopulaParams
                best_AIC = trialAIC
            elif trial_kc_metric <= best_kc and trialAIC < best_AIC and criterion == 'Kc':
                goldCopula = copula
                goldParams = fittedCopulaParams
                best_kc = trial_kc_metric
                best_AIC = trialAIC
        #
        if vb: print("ID: %s. %s copula selected.  fitted params="
                     % (str(self.id), goldCopula.name) + str(goldParams[1])
                     + " rotation=" + str(goldParams[3]))
        if vb: print("-------------------------------------------")
        self.copulaModel = goldCopula
        self.copulaParams = goldParams
        return (self.copulaModel, self.copulaParams)

    def fitCopula(self, copula, thetaGuess=(None, None, )):
        """!
        @brief fit specified copula to data.
        @param copula <b>CopulaBase</b>  Copula instance
        @param thetaGuess <b>tuple</b> (optional) initial guess for copula params
        @return (copula type <b>string</b>, fitted copula params <b>np_array</b>)
        """
        thetaHat, successFlag = \
            copula.fitMLE(self.UU, self.VV, *thetaGuess, weights=self.weights)
        if successFlag:
            AIC = copula._AIC(self.UU, self.VV, 0, *thetaHat, weights=self.weights)
        else:
            AIC = np.inf
        self.copulaModel = copula
        return (copula.name, thetaHat, AIC, copula.rotation, successFlag)

    @staticmethod
    def compute_aic_metric(copula, u, v, rot, theta, weights=None):
        raise NotImplementedError

    def compute_kc_metric(self, copula, t_emp, kc_emp, t_lower=0.4, t_upper=0.6, log=True):
        """!
        @brief Compute l2 norm of differences between the empirical Kc function
        and the fitted copula's Kc function.
        @param copula starvine.bvcopula.copula.copula_base.CopulaBase instance
        @param t_emp <b>np_1darray</b>  emperical kc abcissa
        @param kc_emp <b>np_1darray</b> emperical kc values
        @param t_lower  float. in [0, 1].  Lower t cutoff for comparison
        @param t_upper  float. in [0, 1].  Upper t cutoff for comparison
        """
        import os
        # rotate current data into the proposed copula orientation
        rt_UU, rt_VV = self._rotate_data_bk(self.UU, self.VV, -copula.rotation)
        # compute emperical Kc on the rotated data
        rt_t_emp, rt_kc_emp = self.empKc(rt_UU, rt_VV)
        # fit the un-rotated copula
        base_copula = Copula(copula.name, 0)
        base_copula.fitMLE(rt_UU, rt_VV, *(None, None,), weights=self.weights)
        #
        mask = ((rt_t_emp > 0.01) & (rt_t_emp < 1.0)) # & \
               # ((rt_t_emp < t_lower) | (rt_t_emp > t_upper))
        kc_metric = []
        for i in range(4):
            fitted_kc = base_copula.kC(rt_t_emp[mask])
            kc_metric.append(np.linalg.norm(fitted_kc - rt_kc_emp[mask]))
        if log:
            if not os.path.exists('Kc_logs'):
                os.makedirs('Kc_logs')
            np.savetxt('Kc_logs/kc_log_' + str(base_copula.name) + "_" + str(copula.rotation) + '.txt',
                       np.array([rt_t_emp[mask], fitted_kc, rt_kc_emp[mask]]).T,
                       header="Kendalls fn log for Copula: " + str(base_copula.name) + "_" + str(copula.rotation))
        return np.average(kc_metric)

    def _rotate_data(self, u, v, rotation=0):
        """!
        @brief Rotates the ranked data on the unit square.
        @param u  Ranked data vector
        @param v  Ranked data vector
        @param rotation <b>int</b> 1==90deg, 2==180deg, 3==270, 0==0deg
        @return tuple transposed (u, v) vectors
        """
        UU = np.zeros(u.shape)  # storage for rotated u
        VV = np.zeros(v.shape)  # storage for rotated v
        if self.rotation == 1:
            # 90 degree rotation (flip U)
            UU = 1.0 - u
            VV = v
        elif self.rotation == 2:
            # 180 degree rotation (flip U, flip V)
            UU = 1.0 - u
            VV = 1.0 - v
        elif self.rotation == 3:
            # 270 degree rotation (flip V)
            UU = u
            VV = 1.0 - v
        else:
            UU = u
            VV = v
        return UU, VV

    def _rotate_data_bk(self, u, v, rotation=0):
        assert -3 <= rotation <= 3
        if rotation == -3:
            # -270 deg rotation is eq to +90 deg rotation
            return self._rotate_data(u, v, 1)
        elif rotation == -2:
            # -180 deg rotation is eq to +180 deg rotation
            return self._rotate_data(u, v, 2)
        elif rotation == -1:
            # -90 deg rotation is eq to +270 deg rotation
            return self._rotate_data(u, v, 3)
        elif rotation == 0:
            return u, v
        else:
            return self._rotate_data(u, v, rotation)

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
        if rotation < 0 or rotation > 3:
            print("Invalid Rotation: Valid roations are in [0, 1, 2, 3]")
            raise RuntimeError
        self.rotation = rotation
        self.UU, self.VV = self._rotate_data(self.u, self.v, self.rotation)

    @property
    def defaultFamily(self):
        default_family = {'t': 0,
                          'gauss': 0,
                          'frank': 0,
                          'frank-90': 1,
                          'frank-180': 2,
                          'frank-270': 3,
                          'clayton': 0,
                          'clayton-90': 1,
                          'clayton-180': 2,
                          'clayton-270': 3,
                          'gumbel': 0,
                          'gumbel-90': 1,
                          'gumbel-180': 2,
                          'gumbel-270': 3,
                         }
        return default_family
