# -*- coding: utf-8 -*-
"""Gabor Wavelet Pyramid model for TensorFlow."""
import numpy as np
import skimage.filters
import tensorflow as tf

__author__ = """Johan Carlin"""
__email__ = "johan.carlin@gmail.com"
__version__ = "0.1.0"


class FilterBank(object):
    sigma: float
    cyclespersigma: float
    nsigma: int
    norient: int
    phasevalues: list
    _stride: list
    _filterbank: np.array
    _orientations: np.array
    _filterbank_tensor: []

    def __init__(
        self,
        sigma,
        cyclespersigma=.5,
        nsigma=4,
        norient=8,
        phasevalues=[0, np.pi / 2.],
        stride=[],
    ):
        self.sigma = sigma
        self.cyclespersigma = cyclespersigma
        self.nsigma = nsigma
        self.norient = norient
        self.phasevalues = phasevalues
        self._stride = stride
        self._filterbank = np.array([])
        self._orientations = np.array([])
        self._filterbank_tensor = []
        return

    @property
    def orientations(self):
        if not self._orientations.size:
            self._orientations = np.linspace(0., np.pi, self.norient + 1)[:-1]
        return self._orientations

    @property
    def filterbank(self):
        if not self._filterbank.size:
            self._filterbank = hardstack(
                [
                    gaborweights(
                        frequency=self.cyclespersigma / self.sigma,
                        theta=direction,
                        sigma_x=self.sigma,
                        sigma_y=self.sigma,
                        offset=phase,
                        n_stds=self.nsigma,
                    )
                    for direction in self.orientations
                    for phase in self.phasevalues
                ]
            )
        return self._filterbank

    @property
    def filterbank_tensor(self):
        # nb tf does not suppport 'if not x' syntax so this is necessary
        if self._filterbank_tensor is not None:
            self._filterbank_tensor = tf.convert_to_tensor(
                self.filterbank, dtype="float32"
            )
        return self._filterbank_tensor

    @property
    def stride(self):
        if not self._stride:
            self._stride = [1] + [int(np.ceil(2 * self.sigma))] * 2 + [1]
        return self._stride

    def responseraw(self, x):
        return tf.nn.conv2d(x, self.filterbank_tensor, self.stride, "SAME")

    def responsesimple(self, x):
        resp = self.responseraw(x)
        return tf.nn.relu(resp)

    def responsecomplex(self, x):
        resp = tf.pow(self.responseraw(x), 2)
        # ok so if this was numpy we could just do
        # respshape = tf.reshape(
        #    resp,np.array([1,resp.shape[1],resp.shape[2],self.norient,nphase]))
        # respshape **= 2
        # finalresp = tf.reduce_sum(respshape,axis=-1)
        # but because it's tricky to work with tf arrays of unknown length, it's
        # easier to just sum over each in turn
        # first phase energy

        nphase = len(self.phasevalues)
        finalresp = resp[:, :, :, ::nphase]
        for n in range(nphase - 1):
            # add on each of the further energies
            finalresp = finalresp + resp[:, :, :, (n + 1) :: nphase]
        return tf.sqrt(finalresp)


def gaborweights(*args, **kwargs):
    """return real part of gabor filter, scaled to approximately [-1 1] range,
    and thresholded to remove miniscule weights.
    All input arguments are passed to skimage.filters.gabor_kernel."""
    k = np.real(skimage.filters.gabor_kernel(*args, **kwargs))
    k /= np.abs(np.max(k.ravel()))
    k[np.abs(k) < 0.001] = 0
    return k.astype("float32")


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
