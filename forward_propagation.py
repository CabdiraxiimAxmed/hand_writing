import numpy as np
from activation_function import ReLU, sigmoid

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A, W, b, activation):
    if activation == 'ReLU':
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = ReLU(Z)
    elif activation == 'sigmoid':
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    L = len(parameters) // 2
    caches = []
    A = X

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], 'ReLU')
        caches.append(cache)
    A, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], 'sigmoid')
    caches.append(cache)
    assert(A.shape == (10, X.shape[1]))
    return A, caches
