#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import numpy as np

from .polyfit import *
 
################################################################
# Diffusion Tensor Imaging: GRADIENT DIRECTIONS
################################################################
# --> see dti_funcs.py
################################################################

# some partial derivatives have not yet been implemented,
# the function can be used if the corresponding parameters are
# fixed (ie, the partial derivative is not required);
# setting them to np.NaN should avoid erroneously using them;
# setting them to 0 helps scipy.optimize.minimize(), which does
# not handle fixing of parameters correctly ...

NOT_IMPL = 0 # or np.NaN

################################################################
# fit functions for non-linear IVIM-DTI fitting
################################################################

def sym_matrix(x):
    return np.array([
        [x[0],x[3],x[4]],
        [x[3],x[1],x[5]],
        [x[4],x[5],x[2]]])

def Ds_f_D6(p, x):
    """Ds-f-D6 model, p=9: (S0, d11,d22,d33, d12,d13,d23, f,Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = p[8]
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    return S0 * ((1 - f) * np.exp(-b * diag_gD_gT) + f * np.exp(-b * Ds))

def Ds_f_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = p[8]
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    #
    # S0 * ((1 - f) * np.exp(-b * diag_gD_gT) + f * np.exp(-b * Ds))
    #
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - f) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * Ds)
    f_exp_Ds = f * exp_Ds
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        g[0]*g[0] * mb_S0_onemf_exp_D,
        g[1]*g[1] * mb_S0_onemf_exp_D,
        g[2]*g[2] * mb_S0_onemf_exp_D,
        2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (-exp_D + exp_Ds),
        S0 * f * (-b) * exp_Ds,
    ])

def Ds_f_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],9))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds_f_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],]

################################################################

def Ds6s_f_D6(p, x):
    """Ds6s-f-D6 model, p=9: (S0, d11,d22,d33, d12,d13,d23, f, Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = p[8] * D / p[1:4].mean() # CHECK or sum()?
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
                 + f * np.exp(-b * diag_gDs_gT))

def Ds6s_f_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    trD = p[1:4].sum()
    trD2 = trD**2
    D_tr3 = D / trD * 3 # CHECK or sum()?
    Ds = p[8] * D_tr3
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    #
    # S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
    #             + f * np.exp(-b * diag_gDs_gT))
    #
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDtr3_gT = np.einsum('ij,ij->i', np.dot(g.T,D_tr3), g.T)
    diag_gDs_gT = p[8] * diag_gDtr3_gT
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - f) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = f * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        g[0]*g[0] * mb_S0_onemf_exp_D
        + 3*p[8] * (g[0]*g[0] / trD - diag_gD_gT / trD2) * mb_S0_f_exp_Ds,
        g[1]*g[1] * mb_S0_onemf_exp_D
        + 3*p[8] * (g[1]*g[1] / trD - diag_gD_gT / trD2) * mb_S0_f_exp_Ds,
        g[2]*g[2] * mb_S0_onemf_exp_D
        + 3*p[8] * (g[2]*g[2] / trD - diag_gD_gT / trD2) * mb_S0_f_exp_Ds,
        2 * g[0]*g[1] * (mb_S0_onemf_exp_D + 3*p[8]/trD * mb_S0_f_exp_Ds),
        2 * g[0]*g[2] * (mb_S0_onemf_exp_D + 3*p[8]/trD * mb_S0_f_exp_Ds),
        2 * g[1]*g[2] * (mb_S0_onemf_exp_D + 3*p[8]/trD * mb_S0_f_exp_Ds),
        S0 * (-exp_D + exp_Ds),
        diag_gDtr3_gT * mb_S0_f_exp_Ds,
    ])

def Ds6s_f_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],9))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds6s_f_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],]

################################################################

def Ds_f6s_D6(p, x):
    """Ds-f6s-D6 model, p=9: (S0, d11,d22,d33, d12,d13,d23, f, Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7] * D / p[1:4].mean() # CHECK or sum()?
    Ds = p[8]
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * Ds))

def Ds_f6s_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    trD = p[1:4].sum()
    trD2 = trD**2
    D_tr3 = D / trD * 3 # CHECK or sum()?
    Ds = p[8]
    b = x[:,0]
    g = x[:,1:].transpose()
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDtr3_gT = np.einsum('ij,ij->i', np.dot(g.T,D_tr3), g.T)
    diag_gf_gT = p[7] * diag_gDtr3_gT
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - diag_gf_gT) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * Ds)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (this is the case for segmented fitting)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        S0 * diag_gDtr3_gT * (-exp_D + exp_Ds),
        mb_S0_f_exp_Ds,
    ])

def Ds_f6s_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],9))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds_f6s_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],]

################################################################

def Ds6s_f6s_D6(p, x):
    """Ds6s-f6s-D6 model, p=9: (S0, d11,d22,d33, d12,d13,d23, f, Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7] * D / p[1:4].mean() # CHECK or sum()?
    Ds = p[8] * D / p[1:4].mean() # CHECK or sum()?
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6s_f6s_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    trD = p[1:4].sum()
    trD2 = trD**2
    D_tr3 = D / trD * 3 # CHECK or sum()?
    Ds = p[8] * D_tr3
    b = x[:,0]
    g = x[:,1:].transpose()
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDtr3_gT = np.einsum('ij,ij->i', np.dot(g.T,D_tr3), g.T)
    diag_gDs_gT = p[8] * diag_gDtr3_gT
    diag_gf_gT = p[7] * diag_gDtr3_gT
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - diag_gf_gT) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (this is the case for segmented fitting)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        not_impl,
        S0 * diag_gDtr3_gT * (-exp_D + exp_Ds),
        diag_gDtr3_gT * mb_S0_f_exp_Ds,
    ])

def Ds6s_f6s_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],9))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds6s_f6s_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],]

################################################################

def Ds6a_f_D6(p, x):
    """Ds6a-f-D6 model, p=10: (S0, d11,d22,d33, d12,d13,d23, f, Ds, a), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    a = p[9] # asymmetry factor
    Ds = p[8] * ((1 - a) * np.eye(3) + a * D / p[1:4].mean())
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
                 + f * np.exp(-b * diag_gDs_gT))

def Ds6a_f_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    b = x[:,0]
    g = x[:,1:].transpose()
    S0 = p[0]
    D = sym_matrix(p[1:7])
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDnorm_gT_m1 = diag_gD_gT / p[1:4].mean() - 1
    f = p[7]
    Ds = p[8]
    a = p[9] # asymmetry factor
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * Ds * (1 + a * diag_gDnorm_gT_m1))
    exp_Ds_m_exp_D = exp_Ds - exp_D
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        exp_D + f * exp_Ds_m_exp_D,
        not_impl, #*  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #*  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * exp_Ds_m_exp_D,
        S0 * f * exp_Ds * (1 + a * diag_gDnorm_gT_m1) * (-b),
        S0 * f * exp_Ds * Ds * diag_gDnorm_gT_m1 * (-b),
    ])

def Ds6a_f_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],10))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    p0[:,9] = 0.5
    return p0

Ds6a_f_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],
    [0, 1],]

################################################################

def Ds_f6a_D6(p, x):
    """Ds-f6a-D6 model, p=10: (S0, d11,d22,d33, d12,d13,d23, f, Ds, a), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    a = p[9] # asymmetry factor
    f = p[7] * ((1 - a) * np.eye(3) + a * D / p[1:4].mean())
    Ds = p[8]
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * Ds))

def Ds_f6a_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    b = x[:,0]
    g = x[:,1:].transpose()
    S0 = p[0]
    D = sym_matrix(p[1:7])
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDnorm_gT_m1 = diag_gD_gT / p[1:4].mean() - 1
    f = p[7]
    Ds = p[8]
    a = p[9] # asymmetry factor
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * Ds)
    exp_Ds_m_exp_D = exp_Ds - exp_D
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        exp_D + f * (1 + a*diag_gDnorm_gT_m1) * exp_Ds_m_exp_D,
        not_impl, #*  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #*  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (1 + a*diag_gDnorm_gT_m1) * exp_Ds_m_exp_D,
        S0 * f * (1 + a*diag_gDnorm_gT_m1) * exp_Ds * (-b),
        S0 * f * diag_gDnorm_gT_m1 * exp_Ds_m_exp_D,
    ])

def Ds_f6a_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],10))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    p0[:,9] = 0.5
    return p0

Ds_f6a_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],
    [0, 1],]

################################################################

def Ds6a_f6a_D6(p, x):
    """Ds6a-f6a-D6 model, p=10: (S0, d11,d22,d33, d12,d13,d23, f, Ds, a), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    a = p[9] # asymmetry factor for f and Ds
    f = p[7] * ((1 - a) * np.eye(3) + a * D / p[1:4].mean())
    Ds = p[8] * ((1 - a) * np.eye(3) + a * D / p[1:4].mean())
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6a_f6a_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    b = x[:,0]
    g = x[:,1:].transpose()
    S0 = p[0]
    D = sym_matrix(p[1:7])
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDnorm_gT_m1 = diag_gD_gT / p[1:4].mean() - 1
    DD = diag_gDnorm_gT_m1
    f = p[7]
    Ds = p[8]
    a = p[9] # asymmetry factor for f and Ds
    aDDp1 = 1 + a*DD
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * Ds * aDDp1)
    exp_Ds_m_exp_D = exp_Ds - exp_D
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        exp_D + f * aDDp1 * exp_Ds_m_exp_D,
        not_impl, #*  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #*  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * aDDp1 * exp_Ds_m_exp_D,
        S0 * f * aDDp1 * exp_Ds * (-b) * aDDp1,
        S0 * f * (
            DD * exp_Ds_m_exp_D + aDDp1 * (-b*Ds*DD) * exp_Ds
            ),
    ])

def Ds6a_f6a_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],10))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    p0[:,9] = 0.5
    return p0

Ds6a_f6a_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],
    [0, 1],]

################################################################

def Ds6a_f6ap_D6(p, x):
    """Ds6a-f6a'-D6 model, p=11: (S0, d11,d22,d33, d12,d13,d23, f, Ds, a, a'), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    a_f = p[9] # asymmetry factor for f
    a_Ds = p[10] # asymmetry factor for Ds
    f = p[7] * ((1 - a_f) * np.eye(3) + a_f * D / p[1:4].mean())
    Ds = p[8] * ((1 - a_Ds) * np.eye(3) + a_Ds * D / p[1:4].mean())
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6a_f6ap_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    b = x[:,0]
    g = x[:,1:].transpose()
    S0 = p[0]
    D = sym_matrix(p[1:7])
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDnorm_gT_m1 = diag_gD_gT / p[1:4].mean() - 1
    f = p[7]
    Ds = p[8]
    a_f = p[9] # asymmetry factor for f
    a_Ds = p[10] # asymmetry factor for Ds
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * Ds * (1 + a_Ds*diag_gDnorm_gT_m1))
    exp_Ds_m_exp_D = exp_Ds - exp_D
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        exp_D + f * (1 + a_f*diag_gDnorm_gT_m1) * exp_Ds_m_exp_D,
        not_impl, #*  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #*  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #*  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (1 + a_f*diag_gDnorm_gT_m1) * exp_Ds_m_exp_D,
        S0 * f * (1 + a_f*diag_gDnorm_gT_m1) * exp_Ds * (-b) * (1 + a_Ds*diag_gDnorm_gT_m1),
        S0 * f * diag_gDnorm_gT_m1 * exp_Ds_m_exp_D,
        S0 * f * (1 + a_f*diag_gDnorm_gT_m1) * exp_Ds * (-b*Ds) * diag_gDnorm_gT_m1,
    ])

def Ds6a_f6ap_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],11))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8] = 20 * np.mean(p0[:,1:4], axis=-1)
    p0[:,9] = 0.5
    p0[:,10] = 0.5
    return p0

Ds6a_f6ap_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 100e-3],
    [0, 1],
    [0, 1],]

################################################################

def Ds6e_f_D6(p, x):
    """Ds6e-f-D6 model, p=11: (S0, d11,d22,d33, d12,d13,d23, f, DsE1,DsE2,DsE3), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = sym_matrix_same_eigenvectors(p[8:11], D)
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
                 + f * np.exp(-b * diag_gDs_gT))

def Ds6e_f_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    # Ds = sym_matrix_same_eigenvectors(p[8:11], D)
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    Ds = v @ np.diag(p[8:11]) @ v.T
    b = x[:,0]
    g = x[:,1:].transpose()
    gA = np.dot(g.T, v).T # CHECK
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g.T,Ds), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = f * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        (1 - f) * exp_D + f * exp_Ds,
        not_impl, #  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (exp_Ds - exp_D),
        mb_S0_f_exp_Ds * gA[0]*gA[0],
        mb_S0_f_exp_Ds * gA[1]*gA[1],
        mb_S0_f_exp_Ds * gA[2]*gA[2],
    ])

def Ds6e_f_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],11))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8:11] = 20 * p0[:,1:4]
    return p0

Ds6e_f_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],]

################################################################

def sym_matrix_same_eigenvectors(x, D):
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    # assert: D = v @ np.diag(w) @ v.T
    return v @ np.diag(x) @ v.T

def Ds_f6e_D6(p, x):
    """Ds-f6e-D6 model, p=11: (S0, d11,d22,d33, d12,d13,d23, fE1,fE2,fE3, Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix_same_eigenvectors(p[7:10], D)
    Ds = p[10]
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * Ds))

def Ds_f6e_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    # f = sym_matrix_same_eigenvectors(p[7:10], D)
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    f = v @ np.diag(p[7:10]) @ v.T
    Ds = p[10]
    b = x[:,0]
    g = x[:,1:].transpose()
    gA = np.dot(g.T, v).T # CHECK
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g.T,f), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * Ds)
    S0_exp_Ds_m_exp_D = S0 * (exp_Ds - exp_D)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        (1 - diag_gf_gT) * exp_D + diag_gf_gT * exp_Ds,
        not_impl, #  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0_exp_Ds_m_exp_D * gA[0]*gA[0],
        S0_exp_Ds_m_exp_D * gA[1]*gA[1],
        S0_exp_Ds_m_exp_D * gA[2]*gA[2],
        mb_S0_f_exp_Ds,
    ])

def Ds_f6e_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],11))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7:10] = 0.1
    p0[:,10] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds_f6e_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [0, 100e-3],]

################################################################

def Ds6e_f6e_D6(p, x):
    """Ds6e-f6e-D6 model, p=11: (S0, d11,d22,d33, d12,d13,d23, f, DsE1,DsE2,DsE3), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    Ds = sym_matrix_same_eigenvectors(p[8:11], D)
    Dsmean = max(1e-15, p[8:11].mean())
    f = p[7] * Ds / Dsmean
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6e_f6e_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    # f,Ds = sym_matrix_same_eigenvectors(p[..], D)
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    Ds = v @ np.diag(p[8:11]) @ v.T
    Dsmean = max(1e-15, p[8:11].mean())
    f = p[7] * Ds / Dsmean
    b = x[:,0]
    g = x[:,1:].transpose()
    gA = np.dot(g.T, v).T # CHECK
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g.T,f), g.T)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g.T,Ds), g.T)
    DD = diag_gDs_gT
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = diag_gf_gT * exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        (1 - diag_gf_gT) * exp_D + diag_gf_gT * exp_Ds,
        not_impl, #  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * diag_gDs_gT / Dsmean * (exp_Ds - exp_D),
        S0 * p[7] * (
            ( -1/(3*Dsmean**2) * DD + 1/Dsmean * gA[0]*gA[0] ) * (exp_Ds - exp_D)
             + 1/Dsmean * DD * ((-b)*gA[0]*gA[0]) * exp_Ds
        ),
        S0 * p[7] * (
            ( -1/(3*Dsmean**2) * DD + 1/Dsmean * gA[1]*gA[1] ) * (exp_Ds - exp_D)
             + 1/Dsmean * DD * ((-b)*gA[1]*gA[1]) * exp_Ds
        ),
        S0 * p[7] * (
            ( -1/(3*Dsmean**2) * DD + 1/Dsmean * gA[2]*gA[2] ) * (exp_Ds - exp_D)
             + 1/Dsmean * DD * ((-b)*gA[2]*gA[2]) * exp_Ds
        ),
    ])


def Ds6e_f6e_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],11))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8:11] = 20 * p0[:,1:4]
    return p0

Ds6e_f6e_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],]

################################################################

def Ds6e_f6ep_D6(p, x):
    """Ds6e-f6e'-D6 model, p=13: (S0, d11,d22,d33, d12,d13,d23, fE1,fE2,fE3, DsE1,DsE2,DsE3), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix_same_eigenvectors(p[7:10], D)
    Ds = sym_matrix_same_eigenvectors(p[10:13], D)
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6e_f6ep_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    # Ds = sym_matrix_same_eigenvectors(p[..], D)
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    f = v @ np.diag(p[7:10]) @ v.T
    Ds = v @ np.diag(p[10:13]) @ v.T
    b = x[:,0]
    g = x[:,1:].transpose()
    gA = np.dot(g.T, v).T # CHECK
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g.T,f), g.T)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g.T,Ds), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    # TODO: derivative wrt D tensor components is not yet implemented;
    # this is a bad hack and works only for fits in which the D tensor
    # is not varied, but kept fix (a typical case, admittedly)
    not_impl = NOT_IMPL * np.ones_like(b)
    return -np.array([
        (1 - diag_gf_gT) * exp_D + diag_gf_gT * exp_Ds,
        not_impl, #  g[0]*g[0] * mb_S0_onemf_exp_D,
        not_impl, #  g[1]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  g[2]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        not_impl, #  2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (exp_Ds - exp_D) * gA[0]*gA[0],
        S0 * (exp_Ds - exp_D) * gA[1]*gA[1],
        S0 * (exp_Ds - exp_D) * gA[2]*gA[2],
        mb_S0_f_exp_Ds * gA[0]*gA[0],
        mb_S0_f_exp_Ds * gA[1]*gA[1],
        mb_S0_f_exp_Ds * gA[2]*gA[2],
    ])


def Ds6e_f6ep_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],11))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7:10] = 0.1
    p0[:,10:13] = 20 * p0[:,1:4]
    return p0

Ds6e_f6ep_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [0, 1],
    [0, 1],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],]

################################################################

def Ds6_f_D6(p, x):
    """Ds6-f-D6 model, p=14: (S0, d11,d22,d33, d12,d13,d23, f, Ds11,Ds22,Ds33, Ds12,Ds13,Ds23), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = sym_matrix(p[8:14])
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
                 + f * np.exp(-b * diag_gDs_gT))

def Ds6_f_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = p[7]
    Ds = sym_matrix(p[8:14])
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    #
    # S0 * ((1 - f) * np.exp(-b * diag_gD_gT)
    #             + f * np.exp(-b * diag_gDs_gT))
    #
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g.T,Ds), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - f) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = f * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        g[0]*g[0] * mb_S0_onemf_exp_D,
        g[1]*g[1] * mb_S0_onemf_exp_D,
        g[2]*g[2] * mb_S0_onemf_exp_D,
        2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        S0 * (-exp_D + exp_Ds),
        g[0]*g[0] * mb_S0_f_exp_Ds,
        g[1]*g[1] * mb_S0_f_exp_Ds,
        g[2]*g[2] * mb_S0_f_exp_Ds,
        2 * g[0]*g[1] * mb_S0_f_exp_Ds,
        2 * g[0]*g[2] * mb_S0_f_exp_Ds,
        2 * g[1]*g[2] * mb_S0_f_exp_Ds,
    ])

def Ds6_f_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],14))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7] = 0.1
    p0[:,8:14] = 20 * p0[:,1:7]
    return p0

Ds6_f_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [0, 1],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],]

################################################################

def Ds_f6_D6(p, x):
    """Ds-f6-D6 model, p=14: (S0, d11,d22,d33, d12,d13,d23, f11,f22,f33, f12,f13,f23, Ds), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix(p[7:13])
    Ds = p[13]
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * Ds))

def Ds_f6_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix(p[7:13])
    Ds = p[13]
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    #
    # S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
    #             + diag_gf_gT * np.exp(-b * Ds))
    #
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g.T,f), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - diag_gf_gT) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * Ds)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        g[0]*g[0] * mb_S0_onemf_exp_D,
        g[1]*g[1] * mb_S0_onemf_exp_D,
        g[2]*g[2] * mb_S0_onemf_exp_D,
        2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        g[0]*g[0] * S0 * (-exp_D + exp_Ds),
        g[1]*g[1] * S0 * (-exp_D + exp_Ds),
        g[2]*g[2] * S0 * (-exp_D + exp_Ds),
        2 * g[0]*g[1] * S0 * (-exp_D + exp_Ds),
        2 * g[0]*g[2] * S0 * (-exp_D + exp_Ds),
        2 * g[1]*g[2] * S0 * (-exp_D + exp_Ds),
        mb_S0_f_exp_Ds,
    ])

def Ds_f6_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],14))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7:10] = 0.1
    p0[:,10:13] = 0.01
    p0[:,13] = 20 * np.mean(p0[:,1:4], axis=-1)
    return p0

Ds_f6_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [0, 100e-3],]

################################################################

def Ds6_f6_D6(p, x):
    """Ds6-f6-D6 model, p=19: (S0, d11,d22,d33, d12,d13,d23, f11,f22,f33, f12,f13,f23, Ds11,Ds22,Ds33, Ds12,Ds13,Ds23), x: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix(p[7:13])
    Ds = sym_matrix(p[13:19])
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g,f), g)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g,Ds), g)
    return S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
                 + diag_gf_gT * np.exp(-b * diag_gDs_gT))

def Ds6_f6_D6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    f = sym_matrix(p[7:13])
    Ds = sym_matrix(p[13:19])
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    #
    # S0 * ((1 - diag_gf_gT) * np.exp(-b * diag_gD_gT)
    #             + diag_gf_gT * np.exp(-b * diag_gDs_gT))
    #
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    diag_gf_gT = np.einsum('ij,ij->i', np.dot(g.T,f), g.T)
    diag_gDs_gT = np.einsum('ij,ij->i', np.dot(g.T,Ds), g.T)
    exp_D = np.exp(-b * diag_gD_gT)
    onemf_exp_D = (1 - diag_gf_gT) * exp_D
    mb_S0_onemf_exp_D = (-b) * S0 * onemf_exp_D
    exp_Ds = np.exp(-b * diag_gDs_gT)
    f_exp_Ds = diag_gf_gT * exp_Ds
    mb_S0_f_exp_Ds = (-b) * S0 * f_exp_Ds
    return -np.array([
        onemf_exp_D + f_exp_Ds,
        g[0]*g[0] * mb_S0_onemf_exp_D,
        g[1]*g[1] * mb_S0_onemf_exp_D,
        g[2]*g[2] * mb_S0_onemf_exp_D,
        2 * g[0]*g[1] * mb_S0_onemf_exp_D,
        2 * g[0]*g[2] * mb_S0_onemf_exp_D,
        2 * g[1]*g[2] * mb_S0_onemf_exp_D,
        g[0]*g[0] * S0 * (-exp_D + exp_Ds),
        g[1]*g[1] * S0 * (-exp_D + exp_Ds),
        g[2]*g[2] * S0 * (-exp_D + exp_Ds),
        2 * g[0]*g[1] * S0 * (-exp_D + exp_Ds),
        2 * g[0]*g[2] * S0 * (-exp_D + exp_Ds),
        2 * g[1]*g[2] * S0 * (-exp_D + exp_Ds),
        g[0]*g[0] * mb_S0_f_exp_Ds,
        g[1]*g[1] * mb_S0_f_exp_Ds,
        g[2]*g[2] * mb_S0_f_exp_Ds,
        2 * g[0]*g[1] * mb_S0_f_exp_Ds,
        2 * g[0]*g[2] * mb_S0_f_exp_Ds,
        2 * g[1]*g[2] * mb_S0_f_exp_Ds,
    ])

def Ds6_f6_D6_guess_initial(d):
    # d=(x,y,a) where y.shape=(n_fits, n_x), x.shape=(n_x,)
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],19))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    p0[:,7:10] = 0.1
    p0[:,10:13] = 0.01
    p0[:,13:19] = 20 * p0[:,1:7]
    return p0

Ds6_f6_D6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],
    [-100e-3, 100e-3],]

################################################################
