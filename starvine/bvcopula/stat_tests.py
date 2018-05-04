#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy import random
#
from functools import partial
from multiprocessing import Pool
#
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme, chi2, norm
from scipy.interpolate import interp1d
from numba import jit
# starvine imports
from pc_base import PairCopula


def gauss_copula_test(x1, y1, wgts=None, nboot=8000, dist='ks',
                      alpha=0.05, procs=4, resample=8):
    """!
    @brief Tests if a gaussian copula is a good description of the
    dep structure of a bivaraiate data set.
    @param x1  ndarray, shape (n1, )
    @param y1  ndarray, shape (n1, )
    @param x2 ndarray, shape (n2, )
    @param y2 ndarray, shape (n2, )
    @param dist_metric  str.  in ('ad', 'ks')
    @param alpha float. test significance level
    @return (p_val, d_0, h_dict)
        p_val float.  p-value of test
        d_0 float.    Distance metric
        h_dict dict.  {'h0': Bool}  Result of hypothesis test
    @note Also works with weighted samples by resampling original data
        with replacement.

    let h0 be the hypothesis that the gaussian copula fits the data

    Malevergne, Y. and Sornette, D. Testing the Gaussian Copula
    Hypothesis for Financial Asset Dependencies.
    Quantitative Finance. Vol 3. pp. 231-250, 2001.
    """
    assert nboot >= 80  # require adequate sample size for hypoth test
    if wgts is not None:
        # reasample weighted data with replacement
        pc = PairCopula(x1, y1, weights=wgts, resample=resample)
    else:
        pc = PairCopula(x1, y1)

    # standard normal transform
    y_hat_1 = norm.ppf(pc.UU)
    y_hat_2 = norm.ppf(pc.VV)
    y_hat = np.array([y_hat_1, y_hat_2]).T

    # compute cov matrix and pre-compute inverse
    cov_hat = np.cov(y_hat.T, bias=True)
    cov_hat_inv = np.linalg.inv(cov_hat)
    assert cov_hat_inv.shape == (2, 2,)

    # est orig distance metric
    d_0 = dist_measure(y_hat, cov_hat_inv, dist)
    print("KS-Gauss Dist= %f)" % d_0)

    # estimate p-value by boostrap resampling
    d = np.zeros(nboot)
    if procs > 1:
        pool = Pool(procs)
        d = pool.map(partial(sample_d,
                             cov_hat=cov_hat,
                             cov_hat_inv=cov_hat_inv,
                             dist=dist,
                             N=len(x1)
                            ),
                     range(nboot))
        d = np.array(d)
        pool.close()
    else:
        for i in range(nboot):
            d[i] = sample_d(i, cov_hat, cov_hat_inv, dist, len(x1))
    print("KS-Gauss Empirical Dist Range= (%f, %f))" % (np.min(d), np.max(d)))

    # compute p-val
    # p_val = 1 - d_cdf(d_0)
    p_val = (d >= d_0).sum() / len(d)
    h_dict = {'h0': p_val > alpha}
    return p_val, d_0, h_dict


def sample_d(i, cov_hat, cov_hat_inv, dist, N):
    y_sampled = \
        np.random.multivariate_normal(mean=[0., 0.],
                                      cov=cov_hat, size=N)
    d = dist_measure(y_sampled, cov_hat_inv, dist)
    return d


def dist_measure(y_hat, cov_hat_inv, dist):
    # gen z^2 RV which should be distributed according to a chi-squared
    # distribution if h0 is true (Malevergne 2001)
    z_hat_sqrd = test_z_vector(y_hat, cov_hat_inv)

    # compute empirical CDF of z_hat_sqrd
    F_z_x, F_z_y = ecdf(z_hat_sqrd)

    # dof should be ndim  (pp. 9 in Malevergrne 2001)
    ndim = y_hat.shape[1]
    chi2_frozen = chi2(df=ndim, loc=0., scale=1.0)
    F_z_chi2 = chi2_frozen.cdf(z_hat_sqrd)
    # order lowest to higest (enforce cdf monotone)
    F_z_chi2_ = np.array([z_hat_sqrd, F_z_chi2]).T
    sorted_F_chi2 = F_z_chi2_[F_z_chi2_[:, 0].argsort()]
    F_chi2 = sorted_F_chi2[:, 1]

    # check dims
    assert len(F_z_y) == len(F_chi2)

    # Kolmogorov-Smirnov distance
    dist_map_dict = {'ks': 1, 'ks-avg': 2, 'ad': 3, 'ad-avg': 4}
    dist_int = dist_map_dict[dist]
    d = ks_ad_dist(F_z_y, F_chi2, dist_int)
    return d


@jit(nopython=True)
def ks_ad_dist(F_z_y, F_chi2, dist=1):
    d = 0.0
    if dist == 1:
        d = np.max(np.abs(F_z_y - F_chi2))
    elif dist == 2:
        # more robust to outliers
        d = np.mean(np.abs(F_z_y - F_chi2))
    else:
        numer = np.abs(F_z_y - F_chi2)
        denom = np.sqrt(F_chi2 * (1. - F_chi2))
        if dist == 3:
            d = np.max(numer / denom)
        else:
            # more robust to outliers
            d = np.mean(numer / denom)
    return d


@jit(nopython=True)
def test_z_vector(y_hat, cov_inv):
    """!
    @brief Helper function for dist_measure
    """
    z_hat_sqrd = np.zeros(y_hat.shape[0])
    for k in range(y_hat.shape[0]):
        for i in range(2):
            for j in range(2):
                z_hat_sqrd[k] += y_hat[:, i][k] * cov_inv[i, j] * y_hat[:, j][k]
    return z_hat_sqrd


@jit(nopython=True)
def ecdf(x):
    """!
    @brief Empirical cdf
    @param x np_1darray
    @return np_1darray empirical cdf
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys


def ks2d2s(x1, y1, x2, y2, nboot=None):
    """!
    @brief Two-dimensional Kolmogorov-Smirnov test on two samples.
    @param x1  ndarray, shape (n1, )
    @param y1  ndarray, shape (n1, )
    @param x2 ndarray, shape (n2, )
    @param y2 ndarray, shape (n2, )
    @return tuple of floats (p-val, KS_stat)
        Two-tailed p-value,
        KS statistic

    @note This is the two-sided K-S test. Small p-values means that the two
    samples are significantly different. Note that the p-value is only an
    approximation as the analytic distribution is unkonwn. The approximation is
    accurate enough when N > ~20 and p-value < ~0.20 or so.
    When p-value > 0.20 the value may not be accurate but it implies that the two
    samples are not significantly different. (cf. Press 2007)

    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy,
    Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the
    Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society,
    vol. 225, pp. 155-170 Press, W.H. et al. 2007, Numerical Recipes, section
    14.8
    """
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    return p, D


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


@jit(nopython=True)
def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


@jit(nopython=True)
def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def mardias_test(x, wgts=None, alpha=0.05, cov_bias=False):
    """!
    @brief computes multivariate Mardia's tests for normality.
        Mardia, K. V. (1970), Measures of multivariate skewnees and kurtosis with
        applications. Biometrika, 57(3):519-530.
    @param x  np_2d array with shape = (n_obs, n_dim)
        Each col represents a variable each row is a single
        observation of alll of those vars
    @param wgts observation weights np_1darray with shape = (n_obs,)
        TODO: Does not support weighted samples yet
    @param alpha float. significance level (default == 0.05)
    @param cov_bias bool. argument passed to np.cov for covar matrix normalization
    @return p1, p1c, p2, h_dict
        p1: (float)  skewness test p-val
        p1c: (float)  skewness test p-val adjusted for small samples size ~N < 50
        p2: (float)  kurtosis test p-val
        let h0 be the null hypothesis that the data follows a multivar Gaussian
            hdict: dict of hypothesis test results
             {'alpha': (float) significance level,
              'skew_h0': (bool) if true we can accept h0 wrt. skewness test
              'skew_small_sample_h0': (bool) if true we can accept h0 even if N < 50
              'kurt_h0': (bool) if true we can accept h0 wrt. kurtosis test
              'h0': (bool) if true we can accept h0 wrt skew and kurt
             }
    """
    b1p, b2p, cov = mvar_skew_kurt(x, wgts, cov_bias)
    n, p = x.shape[0], x.shape[1]
    k = ((p + 1) * (n + 1) * (n + 3)) / (n * (((n + 1) * (p + 1.)) - 6))
    # dof of chi2 rv
    dof = (p * (p + 1.) * (p + 2.)) / 6.
    g1c = (n * b1p * k) / 6.
    g1 = (n * b1p) / 6.
    p1 = 1 - chi2.cdf(g1, dof)
    p1c = 1 - chi2.cdf(g1c, dof)
    g2 = (b2p - (p * (p + 2)))/(np.sqrt((8. * p * (p + 2.))/n))
    p2 = 2 * (1 - norm.cdf(abs(g2)))
    # hyothesis result dict
    h_dict = {'alpha': alpha,
              'skew_h0': p1 >= alpha,    # false if skew null hypoth is false
              'skew_small_smaple_h0': p1c >= alpha,
              'kurt_h0': p2 >= alpha,    # false if kurtosis null hypoth is false
              'h0': (p1 > alpha) & (p2 > alpha),  # false if either test fails
              'cov': cov  # covar matrix of data
             }
    return p1, p1c, p2, h_dict


def mvar_skew_kurt(x, wgts=None, cov_bias=False):
    """!
    @brief computes multivariate skewness and kurtosis
    @param x  np_2d array with shape = (n_obs, n_dim)
        Each col represents a variable each row is a single
        observation of all of those vars
    @param cov_bias bool. argument passed to np.cov for covar matrix normalization
        (default is to normalize cov matrix by N-1)
    """
    # compute average vector
    mvar_mu = np.average(x, weights=wgts, axis=0)
    # compute covar matrix
    cov = np.cov(x.T, bias=cov_bias)
    cov_inv = np.linalg.inv(cov)
    # compute multivar skewness
    mvar_skew = (1. / (np.shape(x)[0] ** 2.)) * interior_sum_b1(x, mvar_mu, cov_inv)
    # compute multivar kurtosis
    mvar_kurt = (1 / x.shape[0]) * interior_sum_b2(x, mvar_mu, cov_inv)
    return mvar_skew, mvar_kurt, cov


@jit(nopython=True)
def interior_sum_b1(x, mu, cov_inv):
    """!
    @brief Helper function for mvar_skew_kurt
    """
    sum_b1 = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            sum_b1 += np.dot(x[i, :] - mu, np.dot(cov_inv, (x[j, :] - mu))) ** 3.0
    return sum_b1

@jit(nopython=True)
def interior_sum_b2(x, mu, cov_inv):
    """!
    @brief Helper function for mvar_skew_kurt
    """
    sum_b2 = 0.0
    for i in range(x.shape[0]):
        sum_b2 += np.dot(x[i, :] - mu, np.dot(cov_inv, (x[i, :] - mu))) ** 2.0
    return sum_b2


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    """!
    @breif Energy distance test.
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Energy test by Brian Lau:
        multdist: https://github.com/brian-lau/multdist
    """
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: random.randint(x, size=x)
    else:
        rand = random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        raise NotImplementedError
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z
