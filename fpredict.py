"""fpredict take in a yfinance object and determine the
predictions over a time period.
"""

import os
# switch off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# switch off pandas warnings
pd.set_option('mode.chained_assignment', None)
import tensorflow as tf
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from compiled_functions import create_features, isleak, create_data


INDEX_ZERO = 0
INDEX_ONE = 1


class predict_obj():
    def __init__(self, yf_object):
        self.yf_object = yf_object

    def __repr__(self):
        """Displays the object that has been initialized into the class.

        Returns:
            Str: Object used in the class
        """
        return "<Yahoo Finance object>"

    def write_to_csv(self, name):
        """Write the data from the yfianance pandas dataframe object to a CSV
        file.

        Args:
            name (Str): Name to be used for the file

        Returns:
            Str: Prints the location of where the file has been created.
        """
        object = self.yf_object
        file_path = os.getcwd() + '\\' + name + '.csv'
        object.to_csv(file_path, index=False)
        return f"Data written to {file_path}"

    def graph(self):
        """This graph gives an overview of the stock dataframe for the purpose
        """
        yf_df = self.yf_object
        yf_df['avg'] = yf_df[['Open', 'Close']].mean(axis=1)
        x_axis = yf_df.index
        y_axis = yf_df['avg']
        plt.scatter(x_axis, y_axis, s=5, c='g')
        plt.xlabel('Date')
        plt.ylabel('Avg Cost')
        plt.title('Graph to show the average buy price per date')
        plt.show()

    def predict_test_3(self, no_of_days, code):
        yf_df = self.yf_object
        self.no_of_days = no_of_days
        self.code = code

        # Creates a new dataframe and filters relevant columns
        yf_df = yf_df.reset_index()
        data_df = yf_df[['Date', 'Close']]

        # Changes the index using the Date column and drops the Date column
        # for no duplicates
        data_df.index = data_df['Date']
        data_df = data_df.drop('Date', axis=1)

        # Transform into a 1-D array and scales the data based on the scalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_df)

        # Splitting the data to 2 parts for training and validation
        windows_size = no_of_days*2
        split_sec_one, split_sec_two = create_features(
            scaled_data, windows_size)
        training_data = np.reshape(
            split_sec_one, (split_sec_one.shape[INDEX_ZERO], -1))
        validation = np.reshape(
            split_sec_two, (split_sec_two.shape[INDEX_ZERO], -1))
        # Checks if there is a leak or inconsistent number of data if False
        # means no leak
        print(isleak(scaled_data.shape, split_sec_one.shape,
                     split_sec_two.shape))
        time.sleep(1)

        # First part split is for training data to split x and y
        # NOTE: TO add in a feature to allow the user to specify the
        # no of days of the data that will be used
        x_train, y_train = create_data(
            training_data, no_of_days)

        # reshaping the data to fit the LSTM as LSTM must have a 3-Dimensional
        # data shape (*, *, *)
        x_train = np.reshape(
            x_train, (x_train.shape[INDEX_ZERO], x_train.shape[INDEX_ONE], 1))
        # Creating LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        # mse = mean squared error, optimizer Adam
        model.compile(loss='mse', optimizer='Adam')

        # Epochs number to train...
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

        # NOTE: Saves the model --> need to add a feature for option
        # model.save(f"{code}_model.h5")

        # Takes the total number of values from the split. This is to take a
        # certain number of days for testing a y value in the validation.
        inputs = data_df[len(data_df) -
                         len(validation) - no_of_days:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        # Creates the test array for the number of days
        x_test = []
        for i in range(no_of_days, inputs.shape[0]):
            x_test.append(inputs[i-no_of_days:i, 0])
        x_test = np.array(x_test)

        # Reshapes the data to fit the LSTM model
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_hat = model.predict(x_test)
        predicted_closing_price = scaler.inverse_transform(y_hat)

        split_one = data_df[:windows_size]
        split_two = data_df[windows_size:]
        split_two.loc[:, 'Predicted_Close'] = predicted_closing_price
        plt.plot(split_one['Close'])
        plt.plot(split_two[['Close', 'Predicted_Close']])
        plt.show()
