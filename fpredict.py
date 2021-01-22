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
        rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(y_hat)),2)))
        print(rms)
        plt.plot(yf_df.index, y_actual)
        plt.plot(yf_df.index, y_predicted)
        plt.show()
        # return 

def predict_test_3(self):
    yf_df = self.yf_object
    yf_df = yf_df.reset_index()
    data_df = yf_df[['Date', 'Close']]

    data_df.index = data_df['Date']
    data_df = data_df.drop('Date', axis=1, inplace=True)
    return data_df
