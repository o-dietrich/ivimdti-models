#! /usr/bin/env python3

from __future__ import division, print_function, absolute_import

import numpy as np

from .polyfit import *

################################################################
# DTI - Diffusion Tensor Imaging: GRADIENT DIRECTIONS
################################################################
#
# For this file, we use the .bval/.bvec conventions, in which
# the gradient directions are basically defined within the
# IMAGE REFERENCE SYSTEM. So, x direction is supposed to
# mean going from left to right voxels in acquired image data.
#
################################################################
#
# There are different conventions for the specification of
# diffusion gradient directions:
#
# DICOM (Siemens only or all vendors?)
#
# DICOM gradient directions are provided in physical scanner
# XYZ coordinates! (Cf. the explanation to MRtrix format below.)
#
#
# Step 4. Convert vector to image reference plane
# 
# The vector we retrieved from DICOM header is based on scanner
# reference system. In order to estimate the tensor using FSL Diffusion
# Toolbox, we need to convert it to same frame of references as
# the image. ...
#
# <https://neurohut.blogspot.com/2015/11/how-to-extract-bval-bvec-from-dicom.html>
#
################################################################
# MRtrix format
################################################################
#
# This format consists of a single ASCII text file, with no restrictions
# on the filename. It consists of one row per entry (i.e. per DWI
# volume), with each row consisting of 4 space-separated floating-point
# values; these correspond to [ x y z b ], where [ x y z ] are the
# components of the gradient vector, and b is the b-value in units of
# s/mm². A typical MRtrix format DW gradient table file might look
# like this: grad.b:
#
#          0           0           0           0
#          0           0           0           0
# -0.0509541   0.0617551    -0.99679        3000
#   0.011907    0.955047    0.296216        3000
#  -0.525115    0.839985    0.136671        3000
#  -0.785445     -0.6111  -0.0981447        3000
#   0.060862   -0.456701    0.887536        3000
#   0.398325    0.667699      0.6289        3000
#  -0.680604    0.689645   -0.247324        3000
#   0.237399    0.969995   0.0524565        3000
#   0.697302    0.541873   -0.469195        3000
#  -0.868811    0.407442     0.28135        3000
# ...
#
# It is important to note that in this format, the direction
# vectors are assumed to be provided with respect to real or scanner
# coordinates. This is the same convention as is used in the DICOM
# format. Also note that the file does not need to have the file type
# extension .b (or any other particular suffix); this is simply a
# historical convention.
#
# <https://mrtrix.readthedocs.io/en/latest/concepts/dw_scheme.html>
#
################################################################
# FSL format
################################################################
#
# This format consists of a pair of ASCII text files, typically named
# bvecs & bvals (or variations thereof). The bvals file consists of a
# single row of space-separated floating-point values, all in one row,
# with one value per volume in the DWI dataset. The bvecs file consists
# of 3 rows of space-separated floating-point values, with the first
# row corresponding to the x-component of the DW gradient vectors, one
# value per volume in the dataset; the second row corresponding to the
# y-component, and the third row to the z-component. A typical pair of
# FSL format DW gradient files might look like:
#
# bvecs:
# 0 0 -4.30812931665e-05 -0.00028279245503 -0.528846962834659 -0.78128...
# 0 0 -0.002606397951389 -0.97091525561761 -0.846605326714759  0.61584...
# 0 0 -0.999996760803023  0.23942421337746  0.059831733802001 -0.10168...
#
# bvals:
# 0 0 3000 3000 3000 3000 ...
#
# It is important to note that in this format, the gradient vectors
# are provided with respect to the image axes, not in real or scanner
# coordinates (actually, it’s a little bit more complicated than
# that, refer to the FSL wiki for details). This is a rich source of
# confusion, since seemingly innocuous changes to the image can introduce
# inconsistencies in the b-vectors. For example, simply reformatting the
# image from sagittal to axial will effectively rotate the b-vectors,
# since this operation changes the image axes. It is also important
# to remember that a particular bvals/bvecs pair is only valid for the
# particular image that it corresponds to.
#
# <https://mrtrix.readthedocs.io/en/latest/concepts/dw_scheme.html>
#
#
# What conventions do the bvecs use?
#
# The bvecs use a radiological voxel convention, which is the voxel
# convention that FSL uses internally and originated before NIFTI (i.e.,
# it is the old Analyze convention). If the image has a radiological
# storage orientation (negative determinant of qform/sform) then the
# NIFTI voxels and the radiological voxels are the same. If the image
# has a neurological storage orientation (positive determinant of
# qform/sform) then the NIFTI voxels need to be flipped in the x-axis
# (not the y-axis or z-axis) to obtain radiological voxels. Tools such as
# dcm2niix create bvecs files and images that have consistent conventions
# and are suitable for FSL. Applying fslreorient2std requires permuting
# and/or changing signs of the bvecs components as appropriate, since
# it changes the voxel axes.
#
# <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/FAQ>
#
################################################################






################################################################
# fit functions for non-linear DTI fitting
################################################################

def sym_matrix(x):
    return np.array([
        [x[0],x[3],x[4]],
        [x[3],x[1],x[5]],
        [x[4],x[5],x[2]]])

def dti6(p, x):
    """DTI model, p: (S0, d11,d22,d33, d12,d13,d23), args: (b, g)."""
    S0 = p[0]
    D = sym_matrix(p[1:7])
    b = x[:,0]
    g = x[:,1:]
    # need the diagonal elements of the matrix product g * D * g.T,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g,D), g)
    return S0 * np.exp(-b * diag_gD_gT)

def dti6_derivative(p, d, calc_deriv):
    x, y, a = d # a is not used
    S0 = p[0]
    D = sym_matrix(p[1:7])
    b = x[:,0]
    g = x[:,1:].transpose()
    # need the diagonal elements of the matrix product g.T * D * g,
    # ie, np.diag(np.dot(np.dot(g,D),g.transpose()))
    # this can be calculated most efficiently by using einsum():
    diag_gD_gT = np.einsum('ij,ij->i', np.dot(g.T,D), g.T)
    exp = np.exp(-b * diag_gD_gT)
    mb_S0_exp = (-b) * S0 * exp
    return -np.array([
        exp,
        g[0]*g[0] * mb_S0_exp,
        g[1]*g[1] * mb_S0_exp,
        g[2]*g[2] * mb_S0_exp,
        2 * g[0]*g[1] * mb_S0_exp,
        2 * g[0]*g[2] * mb_S0_exp,
        2 * g[1]*g[2] * mb_S0_exp,
    ])

def dti6_guess_initial(d):
    """d=(x,y) where y.shape=(n_fits, n_x), x.shape=(n_x,)."""
    # handle pre-calc situation!?
    x, y, a = d # a is not used
    # initialize parameters to zero
    p0 = np.zeros((y.shape[0],7))
    # guess p0[:,0] = S0
    p0[:,0] = np.nanmax(y, axis=1)
    # guess p0[:,1] = k
    tmp_y_T = y.copy().T
    tmp_y_T[tmp_y_T < 1] = 0.5
    linfit = polyfit1d(x[:,0], np.log(tmp_y_T), 1)
    p0[:,1:4] = -linfit[1,:,None]
    p0[:,4:7] = 0.1 * p0[:,1:4]
    return p0

dti6_limits = [
    [0, None],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],
    [-5e-3, 5e-3],]

