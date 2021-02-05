"""
fpredict take in a yfinance object and determine the
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
import yfinance as yf
import string
from .model_test import TestMethod
from ..utils import errors
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ..utils.funcs import create_features, isleak, create_data


INDEX_ZERO = 0
INDEX_ONE = 1
RECOMMENDED_NO_OF_DAYS = 60
TIMECODES = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
KEYS = ['longName', 'symbol', 'sector', 'website', 'currency', 'previousClose',
        'regularMarketOpen', 'trailingAnnualDividendYield',
        'regularMarketDayHigh']


class Crawl():
    def __init__(self, code=None,
                 no_of_days=RECOMMENDED_NO_OF_DAYS, filename=None, 
                 model_name=None,
                 timespan=TIMECODES[8],
                 save=False, verbose=0, output_graph=True, filepath=None,
                 epochs=1):
        try:
            # Time span check
            if timespan not in TIMECODES:
                raise errors.YahooFinanceCodeDoesNotExist(code, TIMECODES)
            # Checking Code or Filename option
            if code is not None:
                yf_object = yf.Ticker(code)
                yf_df = yf_object.history(period=timespan)
            elif filename is not None:
                yf_df = pd.read_csv(filename)
            elif filename is not None and code is not None:
                raise ValueError("You can either chooise code or filename.")

            # Initialising items
            self.yf_object = yf_object
            self.yf_df = yf_df
            self.no_of_days = no_of_days
            self.model_name = model_name
            self.timespan = timespan
            self.save = save
            self.verbose = verbose
            self.output_graph = output_graph
            self.filepath = filepath
            self.code = code
            self.epochs = epochs
        except KeyboardInterrupt:
            exit()

    def __repr__(self):
        """Displays the object that has been initialized into the class.

        Returns:
            Str: Object used in the class
        """
        return "<Crawler Object>"

    def model_dev(self):
        self.Test_Method = TestMethod
        self.Test_Method(self.yf_df, self.yf_object, self.code, self.epochs,
                         self.verbose, self.output_graph,
                         self.save, self.filepath,
                         self.no_of_days)

    def model_predict(self, value=2):
        """Predicts data based on saved model

        Args:
            value (int, optional): Number of days to be predicted. Defaults to 1.

        Returns:
            List: Predicted values based on the value
        """
        self.value = value
        # Error need to reconstruct data format take in date and Close only
        reconstructed_model = load_model(self.model_name)
        self.reconstructed_model = reconstructed_model
        df = self.yf_df

        # Creates a new dataframe and filters relevant columns
        yf_df = df.reset_index()
        data_df = yf_df[['Date', 'Close']]
        data_df = data_df.tail(RECOMMENDED_NO_OF_DAYS)

        # Changes the index using the Date column and drops the Date column
        # for no duplicates
        data_df.index = data_df['Date']
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.drop('Date', axis=1)

        # Transform into a 1-D array and scales the data based on the scalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_df)

        inputs = data_df[len(data_df) - RECOMMENDED_NO_OF_DAYS:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        x_data = []
        predicted_data = []
        for index in range(value):
            x_data.append(inputs[(index-int(inputs.shape[0])):
                          int(inputs.shape[0]), 0])
        x_data = np.array(x_data)

        for index in range(value):
            temp_x_train = x_data[index]
            reshaped_scaled_data = np.reshape(temp_x_train, (1,
                                              temp_x_train.shape[INDEX_ZERO],
                                              1))
            yhat = reconstructed_model.predict(reshaped_scaled_data)
            yhat = np.array(yhat)
            yhat = scaler.inverse_transform(yhat)
            print(yhat[0][0])
            predicted_data.append(yhat[0][0])
            if (value - 1) == index:
                break
            else:
                x_data[index+1] = np.append(x_data[index+1], predicted_data)

        newdata_df = pd.DataFrame(predicted_data, columns=["Close"])
        final_data = data_df.append(newdata_df, ignore_index=True)
        plt.title(f"Predictions\nCode: {self.code}")
        plt.plot(final_data[:RECOMMENDED_NO_OF_DAYS], label="Data")
        plt.plot(final_data[RECOMMENDED_NO_OF_DAYS:], label="Predicted")
        plt.xlabel('Index')
        plt.ylabel('Closing Price')
        plt.legend()
        print("Opening Predicted Graph. Close the Graph to Continue.")
        plt.show()
        return f"{predicted_data}"
