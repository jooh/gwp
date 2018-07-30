#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gwp` package."""

# somehow numpy does not get installed when running tests under tox?

import pytest
import numpy as np
import gwp

# TODO - test_gaborweights
# TODO - test_FilterBank

def assert_shape(arrlike, shape):
    np.testing.assert_array_equal(np.array(arrlike).shape, np.array(shape))

def fix_hardstack():
    """fixture for hardstack tests"""
    testshapes = [[5,3], [10,2], [1,5], [1,10]]
    testvals = [np.full(shape, float(scalar)+1.) for scalar, shape in enumerate(testshapes)]
    return gwp.hardstack(testvals)

def test_hardstack_shape():
    assert_shape(fix_hardstack(), [10, 10, 1, 4])
    return

def test_hardstack_vals():
    stack = fix_hardstack()
    np.testing.assert_array_equal(stack[0,4,0,:], np.array([0., 2., 0., 0.]))
    np.testing.assert_array_equal(stack[4,-1,0,:], np.array([0., 0., 0., 4.]))
    np.testing.assert_array_equal(stack[4,4,0,:], np.array([1., 2., 3., 4.]))
    return
