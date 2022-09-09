import numpy as np
import pandas as pd

dataset = pd.read_csv('dataset/mnist_test.csv')

data = np.array(dataset).T
n, m = data.shape
y_test = data[0, :]
x_test = data[1:n]
print(x_test.shape)
