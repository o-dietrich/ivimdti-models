#! /usr/bin/env python3

from __future__ import division
import numpy as np
                    
#######################################################################
# 1D fit of multiple sets of data at once

def polyfit1d(x, y, order, weights=None):
    # need data in columns
    A = np.zeros((len(x), order+1))
    col = 0
    for i in range(order+1):
        A[:,col] = x**i
        col += 1
    if weights is None:
        AT_W = A.T
    else:
        W = np.diag(weights)
        AT_W = np.dot(A.T, W)
    # ORIG: coeffs = (pinv(A'*A) *A' * y);
    coeffs = np.dot(np.dot(np.linalg.pinv(np.dot(AT_W,A)), AT_W), y)
    return coeffs


