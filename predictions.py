import numpy as np
from forward_propagation import L_model_forward

def get_predictions(AL):
    return np.argmax(AL, 0)

def get_accuracy(AL, Y):
    return np.sum(AL == Y) / Y.size

def make_predictions(X, parameters):
    L = len(parameters)
    AL, cache = L_model_forward(X, parameters)
    predictions = get_predictions(AL)
    return predictions

def test_prediction(image, label, parameters):
    predictions = make_predictions(image, parameters)
    print(f'prediction: {predictions}')
    print(f'label: {label}')
