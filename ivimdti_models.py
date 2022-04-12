#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import numpy as np

import ivimdti_nlfit

########################################################################
# auxiliary functions
########################################################################

def sym_matrix(x, UPLO=None):
    if UPLO is not None and UPLO.lower() != 'u' and UPLO.lower() != 'l':
        raise ValueError("UPLO argument must be 'L', 'U', or None")
    z = np.zeros(x.shape[1:] + (3,3))
    z[:,:,:,0,0] = x[0,:]
    z[:,:,:,1,1] = x[1,:]
    z[:,:,:,2,2] = x[2,:]
    if UPLO is None or UPLO.lower() == 'l':
        z[:,:,:,1,0] = x[3,:]
        z[:,:,:,2,0] = x[4,:]
        z[:,:,:,2,1] = x[5,:]
    if UPLO is None or UPLO.lower() == 'u':
        z[:,:,:,0,1] = x[3,:]
        z[:,:,:,0,2] = x[4,:]
        z[:,:,:,1,2] = x[5,:]
    return z

def proc_tensor(x, name='', fn_sym_matrix=sym_matrix):
    if x.shape[-4] != 6:
        raise ValueError("not a linearized tensor; shape: " + str(x.shape))
    D = fn_sym_matrix(x, UPLO='L')
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors
    assert(np.all(w[:,:,:,2] >= w[:,:,:,1]) and
           np.all(w[:,:,:,1] >= w[:,:,:,0])) # sorted eigenvalues returned
    Dmn = np.mean(w, axis=-1, keepdims=True)
    nonzero_mask = np.any(w, axis=-1, keepdims=True)
    FA = np.zeros_like(Dmn)
    FA[nonzero_mask] = np.sqrt(
        1.5 * np.sum((w - Dmn)**2, axis=-1, keepdims=True)[nonzero_mask]
        / np.sum(w**2, axis=-1, keepdims=True)[nonzero_mask])
    RGB = np.zeros_like(w)
    # separate handling for 'cigarre' vs. 'pancake' tensors
    # part 1: cigarre-shaped tensors:
    cigarre_mask = ((w[:,:,:,2] - w[:,:,:,1]) > (w[:,:,:,1] - w[:,:,:,0]))
    max_v = v[cigarre_mask,:,2]
    RGB[cigarre_mask,:] = np.abs(max_v) / \
        np.max(np.abs(max_v), axis=-1, keepdims=True) * FA[cigarre_mask]
    # part 2: pancake-shaped tensors:
    pancake_mask = ((w[:,:,:,2] - w[:,:,:,1]) < (w[:,:,:,1] - w[:,:,:,0]))
    min_v = v[pancake_mask,:,0]
    RGB[pancake_mask,:] = np.abs(min_v) / \
        np.max(np.abs(min_v), axis=-1, keepdims=True) * FA[pancake_mask]
    return {name+'trace': Dmn[:,:,:,0], # remove last (singleton) dimension
            name+'FA': FA[:,:,:,0], # remove last (singleton) dimension
            name+'eigenvalues': w.transpose((3,0,1,2)),
            name+'RGB': RGB,}

def tensor_plus_isotropic(param_list, template_tensor):
    if template_tensor.shape[-4] != 6:
        raise ValueError("not a linearized tensor; shape: "
                         + str(template_tensor.shape))
    tensor_lin = np.zeros_like(template_tensor)
    template_mean = np.mean(template_tensor[:3,:,:,:], axis=0)
    msk = (template_mean != 0)
    tensor_lin[:,msk] = template_tensor[:,msk] / template_mean[msk]
    tensor_lin *= param_list[1]
    tensor_lin[:3,:,:,:] += (1 - param_list[1])
    tensor_lin *= param_list[0]
    return tensor_lin

def tensor_same_eigenvectors(eigenvalue_list, template_tensor,
                             fn_sym_matrix=sym_matrix):
    if template_tensor.shape[-4] != 6:
        raise ValueError("not a linearized tensor; shape: "
                         + str(template_tensor.shape))
    D = fn_sym_matrix(template_tensor, UPLO='L')
    w,v = np.linalg.eigh(D) # w eigenvalues, v eigenvectors (sorted)
    # assert: D = v @ np.diag(w) @ v.T
    # D = np.dot(np.dot(v, D_diag), v.T)
    # return v @ np.diag(eigval_list) @ v.T
    D_diag = np.zeros(template_tensor.shape[1:] + (3,3))
    D_diag[:,:,:,0,0] = eigenvalue_list[0]
    D_diag[:,:,:,1,1] = eigenvalue_list[1]
    D_diag[:,:,:,2,2] = eigenvalue_list[2]
    temp = np.einsum('ijklm,ijkmn->ijkln', v, D_diag)
    tensor = np.einsum('ijklm,ijknm->ijkln', temp, v)
    tensor_lin = np.zeros_like(template_tensor)
    tensor_lin[0,:] = tensor[:,:,:,0,0]
    tensor_lin[1,:] = tensor[:,:,:,1,1]
    tensor_lin[2,:] = tensor[:,:,:,2,2]
    tensor_lin[3,:] = tensor[:,:,:,0,1]
    tensor_lin[4,:] = tensor[:,:,:,0,2]
    tensor_lin[5,:] = tensor[:,:,:,1,2]
    return tensor_lin

def calc_high_b_params(
        b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress):
    b_vals_high = b_vals[b_vals >= b_segm]
    b_vals_grad_high = b_vals_grad[b_vals >= b_segm, :]
    n_b_high = len(b_vals_high)
    diff_high = diff[b_vals >= b_segm, :]
    # bulk-water (high-b) DTI
    p_maps, chi2, niter, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.dti6, b_vals_grad_high, diff_high,
        mask=mask, debug=debug, show_progress=show_progress)
    S01mf = p_maps[0] # S0 without perfusion ("1mf": 1 - f)
    dti = np.zeros((6,)+S01mf.shape)
    for n in range(6):
        dti[n,:] = p_maps[1+n]
    return S01mf, dti, niter

def get_b_vals_grad(b_vals, grads):
    n_b = len(b_vals)
    b_vals_grad = np.zeros((n_b, 4))
    for n in range(n_b):
        b_vals_grad[n,0] = b_vals[n]
        b_vals_grad[n,1:] = grads[:,n]
    return b_vals_grad


########################################################################
# the models:
########################################################################

def prep_p0_from_high_b_Ds_f_D(S0, D):
    p0 = np.zeros(S0.shape + (4,))
    p0[..., 0] = S0 * 1.1
    p0[..., 1] = 0.1
    p0[..., 2] = D
    p0[..., 3] = 20 * D
    return p0
        
def geom_mean(b_vals, diff):
    # NOTE: works only for one particular acquisition/protocol
    # TODO: generalize
    diff_geom = np.zeros((15,)+diff.shape[1:])
    b_vals_geom = np.array((0,5,10,15,20,30,40,50,60,100,200,400,600,800,1000))
    for n,b in enumerate(b_vals_geom):
        msk = (b_vals == b)
        if b > 200:
            for bb in (b-5, b+5):
                msk += (b_vals == bb)
        geommean = np.prod(diff[msk,:,:,:], axis=0)
        msk_nonzero = (geommean > 0.1)
        geommean[msk_nonzero] = geommean[msk_nonzero]**(1/sum(msk))
        diff_geom[n,:,:,:] = geommean
    return b_vals_geom, diff_geom

def Ds_f_D(
        b_vals_orig, diff_orig,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM model: Ds: scalar, f: scalar, D: scalar."""
    # segmentation at b >= b_segm
    b_vals, diff = geom_mean(b_vals_orig, diff_orig)
    # two b-value ranges
    # bulk-water (high-b) ADC
    b_vals_high = b_vals[b_vals >= b_segm]
    n_b_high = len(b_vals_high)
    diff_high = diff[b_vals >= b_segm, :]
    p_maps, chi2, niter1, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.exp_decay, b_vals_high, diff_high,
        mask=mask, debug=debug, show_progress=show_progress)
    S01mf = p_maps[0] # S0 without perfusion ("1mf": 1 - f)
    D = p_maps[1]
    # perfusion (low-b) fitting
    p0_from_diffhigh = prep_p0_from_high_b_Ds_f_D(S01mf, D)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.biexp_decay, b_vals, diff,
        fixed=[False,False,True,False],
        limits=[[0, None], [0, 1], [0, 5e-3], [0, 100e-3]],
        p0=p0_from_diffhigh,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    Ds = p_maps[3] + D # CHECK "+ D" (biexp fits D+Ds in 2nd exp)
    f = p_maps[1]
    b_arr = b_vals_orig.reshape((b_vals_orig.shape[0],1,1,1))
    S = S0 * ((1 - f) * np.exp(-b_arr * D) + f * np.exp(-b_arr * Ds))
    chi2 = np.sum((S - diff_orig)**2, axis=0)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.biexp_decay_limits) # free parameters
    n = len(b_vals_orig)
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return {'S0':S0, 'D':D, 'Ds':Ds, 'f':f,
            'number_of_model_parameters':k,
            'chi2':chi2, 'niter':niter1+niter2, 'aicc':aicc}

########################################################################

def D6(
        b_vals, grads, diff,
        mask=None, p0=None,
        show_progress=True, debug=True):
    """DTI model: D: 3x3 tensor."""
    # S = S0 * exp(-b * gT D g)
    # sort b-values and gradient directions into a single array
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    p_maps, chi2, niter, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.dti6, b_vals_grad, diff, mask=mask, p0=p0,
        debug=debug, show_progress=show_progress)
    S0 = p_maps[0]
    dti = np.zeros((6,)+S0.shape)
    for n in range(6):
        dti[n,:] = p_maps[1+n]
    return_dict = {
        'S0':S0, 'dti':dti,
        'number_of_model_parameters':7,
        'chi2':chi2, 'niter':niter, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds_f_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (9,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0
    
def Ds_f_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: scalar, f: scalar, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds_f_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds_f_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds = p_maps[8] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds_f_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f':f, 'Ds':Ds,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6s_f_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (9,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0

def Ds6s_f_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: D-scaled, f: scalar, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6s_f_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6s_f_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds = p_maps[8] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6s_f_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds':Ds, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds_f6s_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (9,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0
    
def Ds_f6s_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: scalar, f: D-scaled, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds_f6s_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds_f6s_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds = p_maps[8] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds_f6s_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds':Ds, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6s_f6s_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (9,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0
    
def Ds6s_f6s_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: D-scaled, f: D-scaled, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6s_f6s_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6s_f6s_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds = p_maps[8] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6s_f6s_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds':Ds, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6a_f_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (10,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 9] = 0.5
    return p0
    
def Ds6a_f_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: anistropy-scaled, f: scalar, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6a_f_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6a_f_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds_tensor = tensor_plus_isotropic(p_maps[8:10], dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6a_f_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds_tensor':Ds_tensor, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds_f6a_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (10,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 9] = 0.5
    return p0
    
def Ds_f6a_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: scalar, f: anistropy-scaled, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds_f6a_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds_f6a_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    Ds = p_maps[8] 
    f_tensor = tensor_plus_isotropic((p_maps[7],p_maps[9]), dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds_f6a_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds':Ds, 'f_tensor':f_tensor, 
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6a_f6a_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (10,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 9] = 0.5
    return p0

def Ds6a_f6a_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: anistropy-scaled, f: same shape as Ds, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6a_f6a_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6a_f6a_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = tensor_plus_isotropic((p_maps[7],p_maps[9]), dti)
    Ds_tensor = tensor_plus_isotropic((p_maps[8],p_maps[9]), dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6a_f6a_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds_tensor':Ds_tensor,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6a_f6ap_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (11,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 9] = 0.5
    p0[..., 10] = 0.5
    return p0

def Ds6a_f6ap_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: anistropy-scaled, f: a-scaled, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6a_f6ap_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6a_f6ap_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = tensor_plus_isotropic((p_maps[7],p_maps[9]), dti)
    Ds_tensor = tensor_plus_isotropic((p_maps[8],p_maps[10]), dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6a_f6ap_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds_tensor':Ds_tensor,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6e_f_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (11,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    for n in range(3):
        p0[..., 8+n] = 20*dti[n]
    return p0

def Ds6e_f_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: 3 eigenvalues, f: scalar, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6e_f_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6e_f_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds_tensor = tensor_same_eigenvectors(p_maps[8:11], dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6e_f_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds_tensor':Ds_tensor, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds_f6e_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (11,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 0.1
    p0[..., 9] = 0.1
    p0[..., 10] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0

def Ds_f6e_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: scalar, f: 3 eigenvalues, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds_f6e_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds_f6e_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = tensor_same_eigenvectors(p_maps[7:10], dti)
    Ds = p_maps[10] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds_f6e_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds':Ds,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6e_f6e_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (11,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 9] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 10] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0

def Ds6e_f6e_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: 3 eigenvalues, f: same shape as Ds, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6e_f6e_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6e_f6e_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    Ds_tensor = tensor_same_eigenvectors(p_maps[8:11], dti)
    Ds_mean = (p_maps[8] + p_maps[9] + p_maps[10]) / 3
    msk = (Ds_mean != 0)
    f_map_1 = p_maps[7] * p_maps[8]
    f_map_1[msk] /= Ds_mean[msk]
    f_map_2 = p_maps[7] * p_maps[9]
    f_map_2[msk] /= Ds_mean[msk]
    f_map_3 = p_maps[7] * p_maps[10]
    f_map_3[msk] /= Ds_mean[msk]
    f_tensor = tensor_same_eigenvectors((f_map_1,f_map_2,f_map_3), dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6e_f6e_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds_tensor':Ds_tensor,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6e_f6ep_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (13,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 0.1
    p0[..., 9] = 0.1
    p0[..., 10] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 11] = 20 * np.mean(dti[0:3,...], axis=0)
    p0[..., 12] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0

def Ds6e_f6ep_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: 3 eigenvalues, f: 3 eigenvalues, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6e_f6ep_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6e_f6ep_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = tensor_same_eigenvectors(p_maps[7:10], dti)
    Ds_tensor = tensor_same_eigenvectors(p_maps[10:13], dti)
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6e_f6ep_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds_tensor':Ds_tensor,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6_f_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (14,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    for n in range(6):
        p0[..., 8+n] = 20*dti[n]
    return p0
    
def Ds6_f_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: 3x3 tensor, f: scalar, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6_f_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6_f_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f = p_maps[7] 
    Ds_tensor = np.zeros((6,)+S0.shape)
    for n in range(6):
        Ds_tensor[n,:] = p_maps[8+n]
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6_f_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds_tensor':Ds_tensor, 'f':f,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds_f6_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (14,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 0.1
    p0[..., 9] = 0.1
    p0[..., 10] = 0.01
    p0[..., 11] = 0.01
    p0[..., 12] = 0.01
    p0[..., 13] = 20 * np.mean(dti[0:3,...], axis=0)
    return p0
    
def Ds_f6_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: scalar, f: 3x3 tensor, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds_f6_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds_f6_D6, b_vals_grad, diff,
        fixed=[False,True,True,True,True,True,True,
               False,False,False,False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = np.zeros((6,)+S0.shape)
    for n in range(6):
        f_tensor[n,:] = p_maps[7+n]
    Ds = p_maps[13] 
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds_f6_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'f_tensor':f_tensor, 'Ds':Ds,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return return_dict

########################################################################

def prep_p0_from_high_b_Ds6_f6_D6(S0, dti):
    p0 = np.zeros(dti.shape[1:] + (19,))
    if S0.shape == dti.shape:
        S0_tr = np.mean(S0[:3,...], axis=0)
        p0[..., 0] = S0_tr * 1.1
    else:
        p0[..., 0] = S0 * 1.1
    for n in range(6):
        p0[..., 1+n] = dti[n]
    p0[..., 7] = 0.1
    p0[..., 8] = 0.1
    p0[..., 9] = 0.1
    p0[..., 10] = 0.01
    p0[..., 11] = 0.01
    p0[..., 12] = 0.01
    for n in range(6):
        p0[..., 13+n] = 20*dti[n]
    return p0

def Ds6_f6_D6(
        b_vals, grads, diff, S01mf=None, dti=None, niter=None,
        mask=None, b_segm=150,
        show_progress=True, debug=True):
    """IVIM-DTI model: Ds: 3x3 tensor, f: 3x3 tensor, D: 3x3 tensor."""
    n_b = len(b_vals)
    b_vals_grad = get_b_vals_grad(b_vals, grads)
    # two b-value ranges; segmentation at b >= b_segm
    if S01mf is None or dti is None:
        S01mf, dti, niter = calc_high_b_params(
            b_vals, b_vals_grad, diff, mask, b_segm, debug, show_progress)
    p0_from_dti = prep_p0_from_high_b_Ds6_f6_D6(S01mf, dti)
    p_maps, chi2, niter2, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.Ds6_f6_D6, b_vals_grad, diff,
        fixed=[False, True,True,True,True,True,True,
               False,False,False,False,False,False,
               False,False,False,False,False,False],
        p0=p0_from_dti,
        mask=mask, debug=debug, show_progress=show_progress)
    S0 = p_maps[0] # S0 with perfusion
    f_tensor = np.zeros((6,)+S0.shape)
    for n in range(6):
        f_tensor[n,:] = p_maps[7+n]
    Ds_tensor = np.zeros((6,)+S0.shape)
    for n in range(6):
        Ds_tensor[n,:] = p_maps[13+n]
    aicc = np.zeros_like(chi2)
    k = len(ivimdti_nlfit.Ds6_f6_D6_limits) # free parameters
    n = n_b
    aicc[chi2 > 0] = n * np.log(chi2[chi2 > 0]/n) + 2 * (k + 1) \
        + 2 * (k+1) * (k+2) / (n - k - 2)
    return_dict = {
        'S0':S0, 'dti':dti, 'Ds_tensor':Ds_tensor, 'f_tensor':f_tensor,
        'number_of_model_parameters':k,
        'chi2':chi2, 'niter':niter+niter2, 'aicc':aicc}
    return_dict.update(proc_tensor(dti))
    return_dict.update(proc_tensor(Ds_tensor, 'ivim_Ds_'))
    return_dict.update(proc_tensor(f_tensor, 'ivim_f_'))
    return return_dict

########################################################################
