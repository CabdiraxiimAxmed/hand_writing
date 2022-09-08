import numpy as np
from activation_function import sigmoid_deriv, ReLU_deriv
from cost import one_hot

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'ReLU':
        dZ = ReLU_deriv(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_deriv(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    L = len(caches)
    current_cache = caches[L - 1]
    grads = {}

    Y = one_hot(Y) # check here.
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads[f'dA{L - 1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l+1}'], current_cache, 'ReLU')
        grads[f'dA{l}'] = dA_prev_temp
        grads[f'dW{l + 1}'] = dW_temp
        grads[f'db{l + 1}'] = db_temp
    return grads
