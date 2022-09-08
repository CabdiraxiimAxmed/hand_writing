import numpy as np
def initialize_params(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
    return parameters

def update_params(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']
    return parameters
