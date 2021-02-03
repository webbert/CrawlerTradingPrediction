import numpy as np


def create_features(data, window_size):
    """Creates the train test split function where it will split the data based
     on the window_size provided.

    Args:
        data (dataframe or numpy): data to be split
        window_size (float): percentage you want to split it to

    Returns:
        np.array: returns the split sections
    """
    # Splits the data on the number of days wanted to use to train 1 y value.
    # This is determined by the number of days specified * 2.
    split_sec_one = data[0:window_size, 0]
    # Takes the rest of the data
    split_sec_two = data[window_size:, 0]
    return np.array(split_sec_one), np.array(split_sec_two)


def create_data(seg_data, no_of_days):
    """This allows data that has been transformed to be segmented into x and y
    values. The list of values are determined by the number of days you want to
    specify. Based on the number of days, the total scaled data will then be
    split by the index of the range(no_of_days, len(seg_data)) and taking the
    number of days of days specified to get one y value.

    Args:
        seg_data (np.array): Numpy array with segmented data from the original
        scaled_data (np.array): Scaled original data (Transformed)
        no_of_days (int): Number of days to be used x to be used to train one y
        value

    Returns:
        Lists x and y: contains numpy array data.
    """
    x, y = [], []
    # Using the segmentated data, we ill be able to get the start index
    for i in range(no_of_days, len(seg_data)):
        x.append(seg_data[(i-no_of_days):i, 0])
        y.append(seg_data[i, 0])
    x, y = np.array(x), np.array(y)
    return x, y


def isleak(shape, train_shape, test_shape):
    """Checks whether the split sections are equal to one another to prevent
    any leakage of data

    Args:
        shape (numpy shape): Shape of the original dataframe
        train_shape (numpy shape): Shape of the train_shape
        test_shape (numpy shape): shape of the test_shape

    Returns:
        str: True for Leakage, False for no leakage
    """
    leak_result = not(shape[0] == (train_shape[0] + test_shape[0]))
    return f"Check if there is leakge: {leak_result}"
