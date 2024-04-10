#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


def weighted_quantile(x, weights, quantile):
    """
        Computes a given quantile of the data considering
        that each sample has a weight.
        x is a N float array
        weights is a N float array, it expects sum w_i = 1
        quantile is a float in [0.0, 1.0]
    """
    I = np.argsort(x)
    sort_x = x[I]
    sort_w = weights[I]
    acum_w = np.add.accumulate(sort_w)
    norm_w = (acum_w - 0.5*sort_w)/acum_w[-1] 
    interpq = np.searchsorted(norm_w, [quantile])[0] 
    if interpq == 0:
        return sort_x[0]
    elif interpq == len(x):
        return sort_x[-1]
    else:
        tmp1 = (norm_w[interpq] - quantile)/(norm_w[interpq] - norm_w[interpq-1])
        tmp2 = (quantile - norm_w[interpq-1])/(norm_w[interpq] - norm_w[interpq-1])
        assert 0 <= tmp1 <= 1 and 0 <= tmp2 <= 1
        return sort_x[interpq-1]*tmp1 + sort_x[interpq]*tmp2


def wIQR(x, weights):
    """
        Computes the weighted interquartile range (IQR) of x.
        x is a N float array
        weights is a N float array, it expects sum w_i = 1
    """
    return weighted_quantile(x, weights, 0.75) - weighted_quantile(x, weights, 0.25)


def wSTD(x, weights):
    """
        Computes the weighted standard deviation of x.
        x is a N float array
        weights is a N float array, it expects sum w_i = 1
    """
    wmean = np.average(x, weights=weights)
    return np.sqrt(np.average((x - wmean)**2, weights=weights))


def robust_scale(x, weights):
    """
        Computes a robust measure of scale by comparing the weighted versions of the
        standard deviation and the interquartile range of x.
        x is a N float array
        weights is a N float array, it expects sum w_i = 1
    """
    return np.amin([wSTD(x, weights), wIQR(x, weights)/1.349])


def robust_loc(x, weights):
    return weighted_quantile(x, weights, 0.5)
