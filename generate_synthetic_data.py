#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import pathlib
import numpy as np


def build_bvals_grad(bvals, diff_grads):
    bvals_grad = np.zeros((len(bvals) * len(diff_grads), 4))
    n = 0
    for b in bvals:
        for g in diff_grads:
            bvals_grad[n,0] = b
            if b > 0:
                bvals_grad[n,1:] = g / np.sqrt(np.sum(g**2))
            n += 1
    return bvals_grad


def main(base_dir, size=32, SNR=5, debug=False):
    """Generate data, save data."""

    data_dir = base_dir/'TESTDATA'
    pathlib.Path(data_dir).mkdir(exist_ok=True)
    
    bvals = np.array((0,5,10,15,20,30,40,50,60,100,200,400,600,800,1000))
    diff_grads = np.array(((1,0,1),(-1,0,1),(0,1,1),(0,1,-1),(1,1,0),(-1,1,0)))
    bvals_grads = build_bvals_grad(bvals, diff_grads)
    if debug: print(bvals_grads)

    sh = (size,size,size)
    S0 = 10000.0 * np.ones(sh)

    # some linear variation over parameters (trace)
    param_var = np.linspace(0.5,1.5,size)
    # some linear variation over anisotropy (keep trace constant)
    aniso0_var = np.linspace(1.0,1.4,size)
    aniso12_var = np.linspace(1.0,0.8,size)

    # tissue diffusion:
    D = 0.8e-3 * np.ones(sh) * param_var.reshape(size)
    # diagonal elements:
    D0 = D * aniso0_var.reshape((size,1))
    D12 = D * aniso12_var.reshape((size,1))

    # perfusion signal fraction
    f = 0.06 * np.ones(sh) * param_var.reshape((size,1))
    # diagonal elements:
    f0 = f * aniso0_var.reshape((size,1,1))
    f12 = f * aniso12_var.reshape((size,1,1))

    # pseudo-diffusion:
    Ds = 6e-3 * np.ones(sh) * param_var.reshape((size,1,1))
    # diagonal elements:
    Ds0 = Ds * aniso0_var.reshape((size))
    Ds12 = Ds * aniso12_var.reshape((size))

    b = bvals_grads[:,0].reshape((len(bvals_grads),1,1,1))
    g = bvals_grads[:,1:].reshape((len(bvals_grads),3,1,1,1))

    gDg = g[:,0,:,:,:]**2 * D0 + g[:,1,:,:,:]**2 * D12 + g[:,2,:,:,:]**2 * D12
    gfg = g[:,0,:,:,:]**2 * f0 + g[:,1,:,:,:]**2 * f12 + g[:,2,:,:,:]**2 * f12
    gDsg = g[:,0,:,:,:]**2 * Ds0 + g[:,1,:,:,:]**2 * Ds12 + g[:,2,:,:,:]**2 * Ds12
    
    signal = S0 * ((1 - gfg) * np.exp(-b * gDg) + gfg * np.exp(-b * gDsg))

    noise = np.random.standard_normal(sh) + 1j * np.random.standard_normal(sh)
    signal = np.abs(signal + S0.max() / SNR * noise)
    
    if debug: print(signal.shape)
        
    np.save(data_dir/'synthetic_signal.npy', signal)
    np.save(data_dir/'bvals_grads.npy', bvals_grads)

if __name__ == "__main__":

    curdir = pathlib.Path(__file__).parent
    main(curdir)

