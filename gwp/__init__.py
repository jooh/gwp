# -*- coding: utf-8 -*-
"""Gabor Wavelet Pyramid model for TensorFlow."""
import numpy as np
import skimage.filters
import tensorflow as tf

__author__ = """Johan Carlin"""
__email__ = "johan.carlin@gmail.com"
__version__ = "0.1.0"


def convolver(data, filterbank, stride):
    """convenience wrapper around tf conv2d. For critical sampling, set stride=sigma*2"""
    return tf.nn.conv2d(
        data, filterbank, [1] + [int(np.ceil(stride))] * 2 + [1], "SAME"
    )


def n2orientations(norient):
    """return an evenly spaced list of norient orientations (in radians)."""
    return np.linspace(0., np.pi, norient + 1)[:-1]


def gaborbank(sigma, orientations=[0], cyclespersigma=.5, nsigma=4, phase=0):
    """return a 3D array of Gabor filters (vertical * horizontal * orientation)."""
    return hardstack(
        [
            np.real(
                skimage.filters.gabor_kernel(
                    frequency=cyclespersigma / sigma,
                    theta=direction,
                    sigma_x=sigma,
                    sigma_y=sigma,
                    offset=phase,
                    n_stds=nsigma,
                )
            )
            for direction in orientations
        ]
    )


def gaussbank(sigma, orientations=[0], cyclespersigma=.5, nsigma=4):
    """extremely roundabout method to construct 2D Gaussians by generating
    quadrature-offset Gabors with gaborbank and summing over them (see v1energy). Useful
    to ensure that the resulting filters are otherwise identical to the output of
    gaborbank."""
    phasequad = [
        gaborbank(
            sigma,
            orientations=orientations,
            cyclespersigma=cyclespersigma,
            nsigma=nsigma,
            phase=thisphase,
        )
        for thisphase in [0, np.pi / 2]
    ]
    # return only the first 'orientation' channel, since they're all identical after
    # converting the gabors to gaussians (but keep the dim to ensure interchangability
    # with gaborbank)
    return v1energy(*phasequad)[:,:,:,[0]]


def v1energy(*arg):
    """Adelson & Bergen (1985) style rectification by taking the square root of the sum
    of squared phase maps. Each input is assumed to be an activation map of identical
    shape."""
    result = arg[0] ** 2
    for thisphase in arg[1:]:
        result += thisphase ** 2
    # bit ugly but so far we are actually invariant to backend
    sqrter = np.sqrt
    if isinstance(result, tf.Tensor):
        sqrter = tf.sqrt
    return sqrter(result)


def hardstack(k, grayval=0.):
    dim = np.array([thisk.shape for thisk in k])
    # nb ndim is 1-based so this is actually the index for dim end+2
    stackax = k[0].ndim + 1
    newdim = np.max(dim, axis=0)
    for kind, thisk in enumerate(k):
        lpad = np.floor((newdim - np.array(thisk.shape)) / 2).astype(int)
        rpad = np.ceil((newdim - np.array(thisk.shape)) / 2).astype(int)
        finalpad = tuple(zip(lpad, rpad))
        k[kind] = np.pad(thisk, finalpad, "constant", constant_values=grayval)
        k[kind] = np.reshape(k[kind], list(k[kind].shape) + [1])
    return np.stack(k, axis=stackax)
