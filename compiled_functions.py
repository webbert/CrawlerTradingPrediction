import numpy as np

def create_features(data, window_size):
    x_data, y_data = [], []
    for index in range(len(data) - (window_size - 1)):
        window = data[index: (index + window_size), 0]
        x_data.append(window)
        y_data.append(data[index + (window_size - 1), 0])
    return np.array(x_data), np.array(y_data)

def isleak(shape, train_shape, test_shape):
    return not(shape[0] == (train_shape[0] + test_shape[0]))
