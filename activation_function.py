import numpy as np

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_deriv(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)
    return dZ

def ReLU(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def ReLU_deriv(dA, cache): # NOTE: check if anything goes wrong
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ
