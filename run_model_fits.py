#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import time
import pathlib

import numpy as np

import ivimdti_nlfit
import ivimdti_models

########################################################################

def print_result(model, res, duration):
    """Print some results (chi2, AICc) immediately after fitting."""
    print(f"*** {model.__name__}, p={res['number_of_model_parameters']} ***")
    print(f"  time:     {1.0*duration:6.1f} s")
    print("  chi2 (mean, median):  "
          f"{np.mean(res['chi2']):11,.5g}  "
          f"{np.median(res['chi2']):11,.1f}")
    print("  AICc (mean, median):  "
          f"{np.mean(res['aicc']):11.3f}  "
          f"{np.median(res['aicc']):11.3f}")
    print(f"  niter mean:  {np.mean(res['niter']):6.2f}")

########################################################################

def preprocess_segm_dti6(
        b_vals, grads, diff, mask=None, b_segm=150, 
        show_progress=True, debug=False):
    """First step (high-b DTI6) of two-step (segmented) IVIM-DTI6 analysis."""
    # S = S0 * [f * exp(-b * Ds) + (1 - f) * DTImodel]
    n_b = len(b_vals)
    n_fits = diff.size // n_b
    b_vals_grad = np.zeros((n_b, 4))
    for n in range(n_b):
        b_vals_grad[n,0] = b_vals[n]
        b_vals_grad[n,1:] = grads[:,n]
    # two b-value ranges; segmentation at b >= b_segm
    # high-b range
    b_vals_high = b_vals[b_vals >= b_segm]
    b_vals_grad_high = b_vals_grad[b_vals >= b_segm, :]
    n_b_high = len(b_vals_high)
    diff_high = diff[b_vals >= b_segm, :]
    # bulk-water (high-b) DTI
    p_maps, chi2, niter1, aicc = ivimdti_nlfit.image_nonlinear_fit_nda(
        ivimdti_nlfit.dti6, b_vals_grad_high, diff_high,
        mask=mask, debug=debug, show_progress=show_progress)
    S01mf = p_maps[0]
    dti = np.zeros((6,)+p_maps[0].shape)
    for n in range(6):
        dti[n,:] = p_maps[1+n]
    return S01mf, dti, niter1

########################################################################

def ivim_eval(diff, bvals_grad, b_segm=250,
              show_progress=False, debug=False):
    """All 17 performed DTI and IVIM-DTI evaluations."""

    b_vals = bvals_grad[:,0]
    grads = bvals_grad[:,1:].T
    
    res = dict()

    start_time = time.time()
    pre_S01mf, pre_dti6, pre_niter = preprocess_segm_dti6(
        b_vals, grads, diff, b_segm=b_segm,
        show_progress=show_progress, debug=debug)
    print("*** preprocess (high-b) DTI6 ***")
    print("  time: %6.1f s"%(1.0*(time.time() - start_time)))

    # the first two models have different parameters and are called separately
    model = ivimdti_models.Ds_f_D
    start_time = time.time()
    res[model.__name__] = model(
        b_vals, diff, b_segm=b_segm,
        show_progress=show_progress, debug=debug)
    print_result(model, res[model.__name__], time.time() - start_time)

    model = ivimdti_models.D6
    start_time = time.time()
    res[model.__name__] = model(
        b_vals, grads, diff,
        show_progress=show_progress, debug=debug)
    print_result(model, res[model.__name__], time.time() - start_time)

    # all other models are called from this loop
    for model in (
            ivimdti_models.Ds_f_D6,
            ivimdti_models.Ds6s_f_D6,
            ivimdti_models.Ds_f6s_D6,
            ivimdti_models.Ds6s_f6s_D6,
            ivimdti_models.Ds6a_f_D6,
            ivimdti_models.Ds_f6a_D6,
            ivimdti_models.Ds6a_f6a_D6,
            ivimdti_models.Ds6a_f6ap_D6,
            ivimdti_models.Ds6e_f_D6,
            ivimdti_models.Ds_f6e_D6,
            ivimdti_models.Ds6e_f6e_D6,
            ivimdti_models.Ds6e_f6ep_D6,
            ivimdti_models.Ds6_f_D6,
            ivimdti_models.Ds_f6_D6,
            ivimdti_models.Ds6_f6_D6,
        ):
        start_time = time.time()
        res[model.__name__] = model(
            b_vals, grads, diff, S01mf=pre_S01mf, dti=pre_dti6, niter=pre_niter,
            b_segm=b_segm, 
            show_progress=show_progress, debug=debug)
        print_result(model, res[model.__name__], time.time() - start_time)

    return res


#######################################################################

def main(base_dir, b_segm=250, debug=False):
    """Load data, do fitting, save data."""

    data_dir = base_dir/'TESTDATA'
    signal = np.load(data_dir/'synthetic_signal.npy')
    bvals_grads = np.load(data_dir/'bvals_grads.npy')

    if debug: print(bvals_grads)
    
    results = ivim_eval(
        signal, bvals_grads, b_segm=b_segm, debug=debug)

    result_dir = base_dir / f'FIT_RESULTS_b{b_segm}'
    pathlib.Path(result_dir).mkdir(exist_ok=True)
    for model in results:
        n_param = results[model]['number_of_model_parameters']
        for par in results[model]:
            if par == 'number_of_model_parameters':
                continue
            npy_fname = f'fit_{model}_P{n_param}_{par}.npy'
            np.save(result_dir/npy_fname, results[model][par])


if __name__ == "__main__":

    curdir = pathlib.Path(__file__).parent
    main(curdir, b_segm=250)

