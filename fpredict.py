"""fpredict take in a yfinance object and determine the
predictions over a time period.
"""

from compiled_functions import create_features, isleak, create_x_y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from tensorflow.keras import activations
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


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
        yf_df = self.yf_object
        yf_df['avg'] = yf_df[['Open', 'Close']].mean(axis=1)
        x_axis = yf_df.index
        y_axis = yf_df['avg']
        plt.scatter(x_axis, y_axis, s=5, c='g')
        plt.xlabel('Date')
        plt.ylabel('Avg Cost')
        plt.title('Graph to show the average buy price per date')
        plt.show()

    def predict_test_1(self):
        yf_df = self.yf_object
        # create new columns in the data frame that contain data with one day lag.
        yf_df['open'] = yf_df['Open'].shift(1)
        yf_df['high'] = yf_df['High'].shift(1)
        yf_df['low'] = yf_df['Low'].shift(1)
        yf_df['close'] = yf_df['Close'].shift(1)
        # Fills NaN values with Mean and drop the first row
        # Reason for dropping first row due to the shifts
        yf_df = yf_df.fillna(yf_df.mean())
        yf_df = yf_df.drop(yf_df.index[0])
        x_axis = yf_df[['open', 'high', 'low', 'Close']]
        y_axis = yf_df['Close']

        # TrainTestSplit and scale data

        x_train, x_test, y_train, y_test = train_test_split(
            x_axis, y_axis, test_size=0.2, shuffle=False)

        # Kmeans
        kmeans_model = KMeans()
        trained_kmeans = kmeans_model.fit(x_train, y_train)

        # Predict y_hat
        y_hat = trained_kmeans.predict(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_actual = np.concatenate([y_train, y_test])
        y_predicted = np.concatenate([y_train, y_hat])
        # fig, (ax0, ax1) = plt.subplots(ncols=2)
        rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(y_hat)), 2)))
        print(rms)
        plt.plot(yf_df.index, y_actual)
        plt.plot(yf_df.index, y_predicted)
        plt.show()
        # return

    def predict_test_3(self):
        yf_df = self.yf_object
        # Creates a new dataframe
        yf_df = yf_df.reset_index()
        data_df = yf_df[['Date', 'Close']]

        # Changes the index using the Date column and drops the Date column for no duplicates
        data_df.index = data_df['Date']
        data_df = data_df.drop('Date', axis=1)

        # Transform into a 1-D array
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_df)

        # Splitting the data to 2 parts for testing
        # 0.4 = 40% of the data which will be converted to INT
        split_sec_one, split_sec_two = create_features(scaled_data, 0.2)
        split_sec_one = np.reshape(split_sec_one, (split_sec_one.shape[0], -1))
        split_sec_two = np.reshape(split_sec_two, (split_sec_two.shape[0], -1))
        print(split_sec_one)
        # Checks if there is a leak or inconsistent number of data if False means no leak
        print(isleak(scaled_data.shape, split_sec_one.shape, split_sec_two.shape))
        time.sleep(1)

        # First part split is for train_x and x_test
        # NOTE: TO add in a feature to allow the user to specify the
        # window size of the data to be used
        x_train, y_train, start_point = create_x_y(split_sec_one, 0.2)

        # Second part for y_train, x_test
        x_test, y_test = create_x_y(split_sec_two, 0.2)

        # reshaping the data to fit a another dimension
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # Creating LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1,  activation=activations.relu))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)
        model.save("model.h5")


        inputs = data_df[len(data_df) -
                         len(split_sec_two) - start_point:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        x_test = []
        for i in range(start_point, inputs.shape[0]):
            x_test.append(inputs[i-start_point:i, 0])
        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_hat = model.predict(x_train)

        return y_hat
