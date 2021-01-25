import numpy as np


def create_features(data, window_size):
    """Creates the train test split function where it will split the data based on the
    window_size provided.

    Args:
        data (dataframe or numpy): data to be split
        window_size (float): percentage you want to split it to

    Returns:
        [type]: [description]
    """
    # This data will then be minused off by the total to get the index of the position
    index_of_split = int(
        (len(data) - int(len(data) * float(window_size))) - 1)
    # Splits the first section based on the window_size
    # 2nd argument in the split is to retrieve all of the data and fit into one array
    split_sec_one = data[0:index_of_split, 0]
    # Takes the rest of the data
    split_sec_two = data[index_of_split:, 0]
    return np.array(split_sec_one), np.array(split_sec_two)


def create_data(seg_data, scaled_data, size):
    x, y = [], []
    for i in range(60,len(seg_data)):
        x.append(scaled_data[i-60:i,0])
        y.append(scaled_data[i,0])
    x, y = np.array(x), np.array(y)
    return x, y


def isleak(shape, train_shape, test_shape):
    leak_result = not(shape[0] == (train_shape[0] + test_shape[0]))
    return f"Check if there is leakge: {leak_result}"
