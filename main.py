import numpy as np
from predictions import make_predictions, get_accuracy
from get_dataset import x_train, y_train, x_dev, y_dev
from get_test_dataset import x_test, y_test 

parameters = np.load('parameters.npy', allow_pickle = True).tolist()


test_predictions = make_predictions(x_test, parameters)
accuracy = get_accuracy(test_predictions, y_test)
print(accuracy)
