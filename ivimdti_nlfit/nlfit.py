#! /usr/bin/env python3

# NOTE some naming confusion:
#    scipy.optimize.minimize() uses
#       * x and x0 for (fit) parameters and initial values of parameters
#       * args for additional function arguments
#       * fun(x, *args) as function to be minimized (returns chi^2)
#    kapteyn.mpfit-functions use
#       * params (or, here, p) and params0 (p0) for (fit) parameters
#       * x for additional function arguments (such as x-values,
#               where function is to be evaluated)
#       * data and y for data to be fitted

from __future__ import division, print_function, absolute_import

from functools import partial
import numpy as np

from .exp_funcs import *
from .dti_funcs import *
from .ivim_dti_funcs import *


################################################################
# helper and auxiliary functions
################################################################

def akaike_corr(chi2, n, k):
    """Corrected Akaike information criterion (AICc)."""
    # here k ist increased by 1 to (k + 1) everywhere to add the
    # variance (sigma^2) of the data (around the fit function) as
    # an additional model parameter; see, eg:
    # Burnham & Anderson 2004: doi: 10.1177/0049124104268644
    if chi2 > 0 and n > 0 and n - k - 2 > 0:
        AIC = n * np.log(chi2 / n) + 2 * (k + 1)
        AICc = AIC + 2 * (k + 1) * (k + 2) / (n - k - 2)
    else:
        AICc = 0
    return AICc

def residuals_func_template(func, p, d):
    if len(d) == 3:
        x, y, a = d # a is subsequently not used
        return y - func(p, x)
    else:
        x, y, a, aux = d # a is subsequently not used
        return y - func(p, x, aux)

def get_parinfo(n_params, limits, use_deriv, fixed):
    """Return parinfo list for kapteyn fit function."""
    parinfo = list()
    for n in range(n_params):
        lim = ([None, None] if limits == None else limits[n])
        fix = (False if fixed == None else fixed[n])
        if use_deriv == True:
            side = 3
        elif use_deriv == False:
            side = 0
        else:
            side = 3 if use_deriv[n] else 0
        parinfo.append(
            {'limits':lim, 'side':side, 'fixed':fix})
    return parinfo

def p0_to_bounds(p0, bounds):
    """Restrict p0 to range between lower and upper bounds."""
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    l_bounds = np.array([(-np.inf if b is None else b) for b in l_bounds])
    u_bounds = np.array([(np.inf if b is None else b) for b in u_bounds])
    p0 = np.fmax(p0, l_bounds)
    p0 = np.fmin(p0, u_bounds)
    return p0


################################################################
# image post-processing
################################################################

def image_nonlinear_fit_nda(
        func, x, data, aux_data=None, args=None, p0=None, mask=None,
        use_deriv=True, fixed=None, limits=[],
        maxfev=100, chunksize=100, show_progress=False,
        use_kapteyn=True, debug=True):
    """Fit func to image data (for all pixels in mask)."""
    from multiprocessing import Pool
    # functions required for fitting (model, residuals, etc.)
    residuals_func = partial(residuals_func_template, func)
    guess_initial_func = eval(func.__name__+'_guess_initial')
    if limits == []:
        limits = eval(func.__name__+'_limits')
    # parameter processing
    if use_deriv:
        try:
            derivative_func = eval(func.__name__+'_derivative')
        except NameError:
            derivative_func = None
            use_deriv = False
            print(
                "WARNING (ivimdti_nlfit.py:image_nonlinear_fit_nda({:s})): ".format(
                    func.__name__) + "no derivative function available!")
    else:
        derivative_func = None
    if mask is None:
        mask = np.ones(data.shape[1:], dtype=np.bool)
    n_fits = np.sum(mask)
    x = np.array(x)
    n_x = x.shape[0]
    if debug:
        print("n_fits:", n_fits, ", n_x:", n_x, ", x:", x, ", args:", args)
        print("use_deriv:", use_deriv)
    if x.shape[0] != data.shape[0]:
        raise ValueError("incompatible length of data and param")
    # select masked data
    y_masked = np.zeros((n_fits, n_x))
    for n in range(n_x):
        y_masked[:,n] = (data[n,:])[mask]

    # use or guess initial parameters
    if p0 is None:
        p0 = guess_initial_func((x, y_masked, args))
    else: # p0 explicitly given, either same for all fits or for each pixel
        p0 = np.array(p0)
        p0_flat = np.zeros((n_fits, p0.shape[-1]))
        if p0.ndim == 1: # same values for all fits
            for k in range(n_fits):
                p0_flat[k,:] = p0
        else: # individual starting values for each pixel
            for n in range(p0.shape[-1]):
                p0_flat[:,n] = (p0[...,n])[mask]
        p0 = p0_flat
    n_p = p0.shape[-1]
    # prepare static parinfo (for bounds and use of derivative function)
    parinfo = get_parinfo(n_p, limits, use_deriv, fixed)
    p0 = p0_to_bounds(p0, [parinfo[n]['limits'] for n in range(len(parinfo))])

    # prepare pool iterator for parallel processing
    pool = Pool()
    nnan = ~np.isnan(y_masked) # to remove NaN values
    if aux_data is None:
        data_it = ((y_masked[k,nnan[k,:]], x[nnan[k,:]], p0[k,:])
                   for k in range(n_fits))
    else:
        # aux_data has format as data: [N,Z,Y,X], so
        # transpose Z;Y;X to initial positions to apply mask
        # (must transpose before constructing the iterator to
        # avoid computational overhead in parallelized section)
        aux_data_transpose_mask = aux_data.transpose((1,2,3,0))[mask]
        data_it = ((y_masked[k,nnan[k,:]], x[nnan[k,:]], p0[k,:],
                    aux_data_transpose_mask[k,:])
                   for k in range(n_fits))
    if use_kapteyn:
        fit_func = partial(
            kapteyn_fit_func,
            residuals_func=residuals_func, args=args,
            derivative_func=derivative_func,
            parinfo=parinfo, maxfev=maxfev)
    else: # do not use kapteyn_fit_func
        fit_func = partial(
            scipy_fit_func,
            residuals_func=residuals_func, args=args,
            derivative_func=derivative_func,
            parinfo=parinfo, maxfev=maxfev)
    pool_it = pool.imap(fit_func, data_it, chunksize)
    # actual fitting
    p = np.zeros((n_fits, n_p))
    chi2 = np.zeros(n_fits)
    aicc = np.zeros(n_fits)
    niter = np.zeros(n_fits)
    if show_progress:
        print("Fitting progress ...", end="")
    for k in range(n_fits):
        if show_progress and k%100 == 0:
            print("\rFitting progress: %d/%d, %4.1f%%"%(k,n_fits,1e2*k/n_fits),
                  end="", flush=True)
        p[k,:],chi2[k],aicc[k],niter[k] = next(pool_it)
    pool.close()
    if show_progress:
        print("\rFitting progress ... FINISHED" + " "*32)
    # sort data into maps
    p_maps = list()
    for n in range(n_p):
        tmp_map = np.zeros(data.shape[1:])
        tmp_map[mask] = p[:,n]
        p_maps.append(tmp_map)
    chi2_map = np.zeros(data.shape[1:])
    chi2_map[mask] = chi2
    niter_map = np.zeros(data.shape[1:])
    niter_map[mask] = niter
    aicc_map = np.zeros(data.shape[1:])
    aicc_map[mask] = aicc 

    return p_maps, chi2_map, niter_map, aicc_map


################################################################
################################################################

def kapteyn_fit_func(
        data, residuals_func, args=None, derivative_func=None,
        parinfo=[], maxfev=100,):
    """Non-linear least-squares fit to function func."""
    from kapteyn import kmpfit
    if len(data) == 3:
        y, x, p0 = data
        fitobj = kmpfit.Fitter(
            residuals=residuals_func, deriv=derivative_func,
            data=(x,y,args), parinfo=parinfo, maxfev=maxfev)
    else: # have additional aux_data (different for each pixel)
        y, x, p0, aux = data
        fitobj = kmpfit.Fitter(
            residuals=residuals_func, deriv=derivative_func,
            data=(x,y,args,aux), parinfo=parinfo, maxfev=maxfev)
    # uncomment this for debugging (the exception handling
    # hides the errors ...)
    #fitobj.fit(params0=p0)
    try:
        fitobj.fit(params0=p0)
    except (RuntimeError, ValueError, SystemError) as e:
        # Non-finite parameter from mpfit.c
        print("EXCEPTION (ivimdti_nlfit.py:fitobj.fit(): ", e)
        print("current initial parameters p0:", p0)
    return (fitobj.params, fitobj.chi2_min,
            akaike_corr(fitobj.chi2_min, len(x), len(p0)), fitobj.niter)

################################################################

# the non-kapteyn alternative is at least sort of working
# (with L-BFGS-B or perhaps TNC,
# but substantially slower than the kapteyn variant,
# and minimization does often not work well)

def scipy_fit_func_chi2(p, d, residuals_func, scale=1):
    p_orig_scaled = p * scale
    return np.sum(residuals_func(p_orig_scaled, d)**2)

def scipy_fit_func_deriv(p, d, residuals_func, derivative_func, scale=1):
    p_orig_scaled = p * scale
    gradients = derivative_func(p_orig_scaled, d, None)
    residuals = residuals_func(p_orig_scaled, d)
    return scale * np.sum(2*residuals * gradients, axis=1)

def scipy_fit_bounds_from_parinfo(parinfo, p0):
    bounds = list()
    for n in range(len(parinfo)):
        if not parinfo[n]['fixed']:
            bounds.append(parinfo[n]['limits'].copy())
        else:
            # this is hacky to suppress fitting of parameter #n
            bounds.append([p0[n],p0[n]])
    return bounds

def scipy_fit_func(
        data, residuals_func, args=None, derivative_func=None,
        parinfo=[], maxfev=100,
        method='L-BFGS-B', # 'TNC' or 'L-BFGS-B',
        apply_scaling=True, # or apply_scaling=False
    ):
    """Non-linear least-squares fit to function func."""
    # available methods: T-BFGS-B, TNC
    # not working well or too slow: Nelder-Mead, Powell, CG, BFGS, COBYLA
    import scipy.optimize
    if len(data) == 3:
        y, x, p0 = data
    else:
        y, x, p0, aux = data
    bounds = scipy_fit_bounds_from_parinfo(parinfo, p0)
    # scaling of the parameters to the order of 1 helps some
    # minimizing algorithms, so we scale either by the initial
    # values (if != 0) or by the magnitude of the bounds (if != inf)
    # or by 1
    scale = np.ones_like(p0)
    if apply_scaling:
        for n in range(len(p0)):
            if p0[n] != 0:
                scale[n] = np.abs(p0[n])
            elif bounds[n][1] is not None and bounds[n][1] != 0:
                scale[n] = np.abs(bounds[n][1]/2)
            if bounds[n][0] is not None:
                bounds[n][0] = bounds[n][0] / scale[n]
            if bounds[n][1] is not None:
                bounds[n][1] = bounds[n][1] / scale[n]
    p0_new_scale = p0 / scale
    chi2_func = partial(
        scipy_fit_func_chi2, residuals_func=residuals_func, scale=scale)
    if derivative_func is not None:
        jac = partial(scipy_fit_func_deriv, residuals_func=residuals_func,
                      derivative_func=derivative_func, scale=scale)
    else:
        jac = False
    
    if method == 'TNC':
        options = {'maxiter': maxfev}
    elif method == 'L-BFGS-B':
        options = {'maxfun': maxfev}
    else:
        options = {}
        bounds = None
    if len(data) == 3:
        fit_result = scipy.optimize.minimize(
            chi2_func, p0_new_scale, args=((x,y,args),),
            method=method, bounds=bounds, jac=jac, options=options,
        )
    else:
        fit_result = scipy.optimize.minimize(
            chi2_func, p0_new_scale, args=((x,y,args,aux),),
            method=method, bounds=bounds, jac=jac, options=options,
        )
    p = fit_result.x * scale
    chi2 = fit_result.fun
    try:
        niter = fit_result.nit
    except AttributeError:
        niter = -1
    return (p, chi2, akaike_corr(chi2, len(x), len(p0)), niter)

