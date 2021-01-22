"""fpredict take in a yfinance object and determine the
predictions over a time period.
"""

from compiled_functions import create_features, isleak
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
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
        return "<Yahoo Finance object>"

    def write_to_csv(self, name):
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
        plt.plot(yf_df.index, y_actual)
        plt.plot(yf_df.index, y_predicted)
        plt.show()
        # return 

    def predict_test_2(self):
        yf_df = self.yf_object
        yf_df = yf_df.reset_index()
        yf_df = yf_df[['Date', 'Close']]

        new_df = yf_df.loc[0: 199]
        new_df = new_df.drop('Date', axis=1)
        new_df = new_df.reset_index(drop=True)

        # reshape index 0 column value -1 One shape dimension can be -1., index 1 row
        new_df = np.reshape(new_df.values, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_array = scaler.fit_transform(new_df)

        # train test split size
        train_size = int(len(scaled_array) * 0.8)
        test_size = int(len(scaled_array) - train_size)
        train, test = scaled_array[0:train_size], scaled_array[train_size:len(
            scaled_array)]

        # Window size = number of days
        window_size = 20

        # train test split function
        x_train, y_train = create_features(train, window_size)
        x_test, y_test = create_features(test, window_size)
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        # print(x_test.shape, x_train.shape)
        scaled_array_shape = scaled_array.shape
        train_shape = train.shape
        test_shape = test.shape

        # Check whether the data is equal and no leakage
        leak_res = isleak(scaled_array_shape, train_shape, test_shape)
        if leak_res is False:
            print('No leakage..')
        else:
            print('Leakage present.')

        tf.random.set_seed(11)
        np.random.seed(11)

        model = Sequential()

        # Units = 50 --> 50 neurons, relu learn from non-lineralities
        model.add(LSTM(units=50, activation='relu',
                       input_shape=(x_train.shape[0], window_size)))

        # turn off 20% of neurons, prevents overfitting
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer='adam')

        filepath = os.getcwd() + "\\" + "test.hdf5"

        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        model.fit(x_train, y_train, epochs=100, batch_size=20, validation_data=(x_test, y_test),
                  callbacks=[checkpoint],
                  verbose=0, shuffle=False)

        best_model = load_model('test.hdf5')

        y_hat = best_model.predict(x_train)

        y_hat_train = scaler.inverse_transform(y_hat)

        y_hat_test_predict = best_model.predict(x_test)

        y_hat_test = scaler.inverse_transform(y_hat_test_predict)

        y_actual = np.concatenate((y_train, y_test))
        y_predict = np.concatenate((y_hat_train, y_hat_test))
        plt.plot(y_predict)
        plt.plot(y_actual)
        plt.show()
