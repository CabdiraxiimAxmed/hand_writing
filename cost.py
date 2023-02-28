import numpy as np

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#   print(one_hot_Y.shape)
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def compute_cost(AL, Y):
    Y = one_hot(Y)
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost
