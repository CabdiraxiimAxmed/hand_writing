import numpy as np
import pandas as pd

data = pd.read_csv('dataset/mnist_train.csv')
data = np.array(data)
m, n = data.shape
data_dev = data[0:1000].T
x_dev = data_dev[1:n]
y_dev = data_dev[0, :]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0, :]
x_train = data_train[1:n]
x_train = x_train / 255.
