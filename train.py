import numpy as np
import pandas as pd
from initialize_params import initialize_params, update_params
from cost import compute_cost
from forward_propagation import L_model_forward
from backward_propagation import L_model_backward
from get_dataset import x_train, y_train, x_dev, y_dev
from predictions import make_predictions, test_prediction, get_accuracy

def nn_model(X, Y, layers_dims, num_iterations, learning_rate = 0.10, print_cost = True):
    parameters = initialize_params(layers_dims)
    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_params(parameters, grads, learning_rate)
        print(f'{i}: {cost}')
        if print_cost and i%100 == 0:
            costs.append(cost)
    return parameters

layers_dims = [784, 10, 10]
# parameters = initialize_params(layers_dims)
parameters = nn_model(x_train, y_train, layers_dims, 1000)


for i in range(200):
    image = x_train[:, i, None]
    label = y_train[i]
    result = test_prediction(image, label, parameters)

dev_predictions = make_predictions(x_dev, parameters)
accuracy = get_accuracy(dev_predictions, y_dev)

print(accuracy)
np.save('parameters.npy', parameters)
