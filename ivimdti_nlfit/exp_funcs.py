#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import numpy as np

from .polyfit import *

################################################################
# fit functions for non-linear fitting
################################################################

def exp_decay(p, x):
    """exponential decay S0 * exp(-k*x), p = (S0, k)."""
    y = np.exp(-p[1] * x)
    y *= p[0]
    return y

def exp_decay_derivative(p, d, calc_deriv):
    x, y, a = d
    exp = np.exp(-p[1] * x)
    deriv = -np.array([
        exp,
        (-x) * p[0] * exp,
    ])
    return deriv

def exp_decay_guess_initial(d):
    """d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)."""
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],2))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    res = polyfit1d(x, np.log(tmp_y_T), 1)
    p0[:,1] = -res[1,:]
    return p0

exp_decay_limits = [
    [0,None],
    [0,None],]

################################################################

def biexp_decay(p, x):
    """bi-expon. decay S0*[(1-f)*exp(-k1*x)+f*exp(-(k1+k2)*x)], p = (S0,f,k1,k2)."""
    y = (1-p[1]) * np.exp(-p[2] * x)
    y += p[1] * np.exp(-(p[2]+p[3]) * x)
    y *= p[0]
    return y

def biexp_decay_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    signal1 = np.exp(-(p[2]+p[3]) * x)
    signal2 = np.exp(-p[2] * x)
    signal = p[1] * signal1 + (1 - p[1]) * signal2
    deriv = np.array([
        signal,
        p[0] * (signal1 - signal2),
        p[0] * (-x) * signal,
        p[0] * (-x) * p[1] * signal1,
    ])
    return -deriv # minus b/o difference in residual

def biexp_decay_guess_initial(d):
    """d=(x,y) where y.shape=(n_fits, n_x), x.shape=(n_x,)."""
    # p = (S0,f,k1,k2)
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],4))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = f
    p0[:,1] = 0.2
    # guess p0[:,2] = k1 (slow)
    dy = p0[:,0] - np.nanmean(y, axis=1)
    ydx = p0[:,0] * (np.nanmean(x) - np.nanmin(x))
    ydx[ydx < 1e-10] = 1 # CHECK: any better value??
    p0[:,2] = dy / ydx
    # guess p0[:,3] = k2 (fast)
    p0[:,3] = 5 * p0[:,2]
    return p0

biexp_decay_limits = [
    [0, None],
    [0, 1],
    [0, None],
    [0, None],]

################################################################
