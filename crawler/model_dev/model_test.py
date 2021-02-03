import os
# switch off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from ..utils.funcs import create_features, isleak, create_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from ..utils import errors
import matplotlib.pyplot as plt
import numpy as np
import string
import tensorflow as tf


INDEX_ZERO = 0
INDEX_ONE = 1
RECOMMENDED_NO_OF_DAYS = 60
KEYS = ['longName', 'symbol', 'sector', 'website', 'currency', 'previousClose',
        'regularMarketOpen', 'trailingAnnualDividendYield',
        'regularMarketDayHigh']


class TestMethod():
    def __init__(self, yf_df, code, epochs, verbose, output_graph, save, filepath,
                 no_of_days):
        self.yf_df = yf_df
        self.code = code
        self.epochs = epochs
        self.verbose = verbose
        self.output_graph = output_graph
        self.save = save
        self.filepath = filepath
        self.no_of_days = no_of_days
        self.model_dev()

    def __repr__(self):
        return "<Test Model Object>"

    def info(self):
        print("Printing Relevant Information..")
        yf = self.yf_object
        information = yf.info
        for key in KEYS:
            print(f"{key.capitalize()}: {information[key]}")

    def view_graph(self, df, windows_size, predicted, code):
        self.info()
        print("Creating Visual Graph...")
        split_one = df[:windows_size]
        split_two = df[windows_size:]
        split_two.loc[:, 'Predicted_Close'] = predicted
        plt.title(f"Testing model with actual and predicted data\nCode "
                  f": {code}")
        plt.plot(split_one['Close'])
        plt.plot(split_two['Close'], label="Actual")
        plt.plot(split_two['Predicted_Close'], label="Predicted")
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        print("Opening Graph. Close the Graph to Continue.")
        plt.legend()
        plt.show()

    def model_dev(self):
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

        # Splitting the data to 2 parts for training and validation
        windows_size = no_of_days*2
        split_sec_one, split_sec_two = create_features(
            scaled_data, windows_size)

        # Checking to see whether the number of days can be used to train
        if windows_size > (len(split_sec_one) or len(split_sec_two)):
            raise errors.DaysInputError(no_of_days, len(data_df))

        training_data = np.reshape(
            split_sec_one, (split_sec_one.shape[INDEX_ZERO], -1))
        validation = np.reshape(
            split_sec_two, (split_sec_two.shape[INDEX_ZERO], -1))
        # Checks if there is a leak or inconsistent number of data if False
        # means no leak
        print(isleak(scaled_data.shape, split_sec_one.shape,
                     split_sec_two.shape))

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
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=1,
                  verbose=self.verbose)

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

        # print graph
        if self.output_graph:
            self.view_graph(data_df, windows_size, predicted_closing_price,
                            self.code)

        # Saves the model
        if self.save and self.filepath is None:
            current_filepath = os.getcwd() + f"\\{self.code}_model"
            model.save(current_filepath)
            print(f"Saved model at {current_filepath}.")
        elif self.save and self.filepath is not None:
            model.save(self.filepath)
            print(f"Saved model at {self.filepath}.")
        return self.__repr__
