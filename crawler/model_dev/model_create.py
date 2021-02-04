import os
# switch off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from ..utils.funcs import create_features, isleak, create_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from ..utils import errors
import numpy as np
import string
import tensorflow as tf
import pathlib


INDEX_ZERO = 0
INDEX_ONE = 1
RECOMMENDED_NO_OF_DAYS = 60


class CreateModel():
    def __init__(self, yf_df, code, filepath, no_of_days, epochs, verbose=0):
        self.yf_df = yf_df
        self.code = code
        self.filepath = filepath
        self.no_of_days = no_of_days
        self.epochs = epochs
        self.verbose = verbose
        self.model_create()

    def model_create(self):
        yf_df = self.yf_df
        no_of_days = self.no_of_days

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
        train_data = np.reshape(
            scaled_data, (scaled_data.shape[INDEX_ZERO], -1))
        x_train, y_train = create_data(
            train_data, no_of_days)
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

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=1,
                  verbose=self.verbose)

        if self.filepath is None:
            current_filepath = (str(pathlib.Path(__file__).parent.absolute()) +
                                f"\\{self.code}_model")
            model.save(current_filepath)
            print(f"Saved model at {current_filepath}.")
        elif self.filepath is not None:
            current_filepath = self.filepath + f"\\{self.code}_model"
            model.save(current_filepath)
            print(f"Saved model at {current_filepath}.")
