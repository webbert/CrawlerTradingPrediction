# Side Project

Coming up with a trading bot and predicting the future

## Plan

1. First step is to use machine learning based on the data to try to analyse and whether machine learning can predict the future

2. Use an API such as alpha vantage and determine whther it is possible to use a trading strategy and build a model

3. Test it out on test data first.

## Progress Day 1 (13 Jan 2021)

Yfinance API <https://pypi.org/project/yfinance/>

Alphavantage Documentation <https://www.alphavantage.co/documentation/>

Trading Strategies:

<https://www.investopedia.com/articles/trading/06/daytradingretail.asp>

<https://www.investopedia.com/articles/active-trading/090415/only-take-trade-if-it-passes-5step-test.asp>

Reits:
Lendlease <https://sg.finance.yahoo.com/quote/JYEU.SI?p=JYEU.SI&.tsrc=fin-srch>

## Progress day 2 (14 Jan 2021)

How to create trading prediction <https://blog.quantinsti.com/trading-using-machine-learning-python/#prereq>

Randomised Search CV for Hyperparameter <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>

### MODEL USED FOR PREDICTION

Lasso <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>

Lasso : Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean.

Parmameters:
{'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': 1000, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': None, 'selection': 'cyclic', 'tol': 0.0001, 'warm_start': False}

In randomised search cv, CV is the number of cross validations. In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as gtraining data.

The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. Cross-validation combines (averages) measures of fit (prediction error) to derive a more accurate estimate of model prediction performance.

## Progress Day 3 (18 Jan)

<https://medium.com/auquan/https-medium-com-auquan-machine-learning-techniques-trading-b7120cee4f05>

RMSE (ROOT MEAN SQUARED ERROR) = √(Σ(y_actual - y_predicted))/number of samples

TO DO: Fit transform. Different shape as seen in the x_train and y_train.

## Progress Day 4 (22 Jan)

Currently creating a LTSM machine learning model for prediction.

Length of scaled data = 3316
We take approx 20 % of the data as testing.
Followed by that we use 80% as training.
<https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/>

## Progress Day 5 (25 Jan)

Changed the code for the predicted test.
Currently working well.

Need to format the code and understand the splitting functions and the arrays.

![Image of the test done for Prediction of Stocks](\images/lendlease_prediction.png)

## Progress Day 6 (26 Jan)

Bugs Found:

1. Noticed that if the data exceeds a certain number of days based on the split section, it cannot run.

Need to understand the logic for the inputs areas.

Made modifications to the predict method in the class.

__TO DO:__

To create more efficient code layout and start making a plan for the project

## Progress Day 7 (28 Jan)

Made some reformatting to the main structure of the project.

Want to learn how to make code and coding structure more professional and efficient.

## Progress Day 8 (29 Jan)

Created version number for product

Bugs Found:

1. Noticed that if the data exceeds a certain number of days based on the split section, it cannot run.

- Need to find solution to be able to fit number of days and windows size

## Progress Day 9 (30 Jan)

Different Periods and intervals and parameters

1. Period: data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
2. Interval: data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
3. start: If not using period - Download start date string (YYYY-MM-DD) or datetime.
4. end: If not using period - Download end date string (YYYY-MM-DD) or datetime
5. prepost: Include Pre and Post market data in results? (Default is False
6. auto_adjust: Adjust all OHLC automatically? (Default is True)
7. actions: Download stock dividends and stock splits events? (Default is True)

Keys for Information:

dict_keys(['zip', '**sector**', 'fullTimeEmployees', 'longBusinessSummary', 'city', 'phone', 'country', 'companyOfficers', '**website**', 'maxAge', 'address1', 'industry', 'address2', '**previousClose**', '**regularMarketOpen**', '**twoHundredDayAverage**', '**trailingAnnualDividendYield**', 'payoutRatio', 'volume24Hr', '**regularMarketDayHigh**', 'navPrice', 'averageDailyVolume10Day', 'totalAssets', 'regularMarketPreviousClose', 'fiftyDayAverage', 'trailingAnnualDividendRate', 'open', 'toCurrency', 'averageVolume10days', 'expireDate', 'yield', 'algorithm', 'dividendRate', 'exDividendDate', 'beta', 'circulatingSupply', 'startDate', 'regularMarketDayLow', 'priceHint', '**currency**', 'regularMarketVolume', 'lastMarket', 'maxSupply', 'openInterest', 'marketCap', 'volumeAllCurrencies', 'strikePrice', 'averageVolume', 'priceToSalesTrailing12Months', 'dayLow', 'ask', 'ytdReturn', 'askSize', 'volume', 'fiftyTwoWeekHigh', 'forwardPE', 'fromCurrency', 'fiveYearAvgDividendYield', 'fiftyTwoWeekLow', 'bid', 'tradeable', 'dividendYield', 'bidSize', 'dayHigh', 'exchange', 'shortName', '**longName**', 'exchangeTimezoneName', 'exchangeTimezoneShortName', 'isEsgPopulated', 'gmtOffSetMilliseconds', 'quoteType', 'symbol', 'messageBoardId', 'market', 'annualHoldingsTurnover', 'enterpriseToRevenue', 'beta3Year', 'profitMargins', 'enterpriseToEbitda', '52WeekChange', 'morningStarRiskRating', 'forwardEps', 'revenueQuarterlyGrowth', 'sharesOutstanding', 'fundInceptionDate', 'annualReportExpenseRatio', 'bookValue', 'sharesShort', 'sharesPercentSharesOut', 'fundFamily', 'lastFiscalYearEnd', 'heldPercentInstitutions', 'netIncomeToCommon', 'trailingEps', 'lastDividendValue', 'SandP52WeekChange', 'priceToBook', 'heldPercentInsiders', 'nextFiscalYearEnd', 'mostRecentQuarter', 'shortRatio', 'sharesShortPreviousMonthDate', 'floatShares', 'enterpriseValue', 'threeYearAverageReturn', 'lastSplitDate', 'lastSplitFactor', 'legalType', 'lastDividendDate', 'morningStarOverallRating', 'earningsQuarterlyGrowth', 'dateShortInterest', 'pegRatio', 'lastCapGain', 'shortPercentOfFloat', 'sharesShortPriorMonth', 'impliedSharesOutstanding', 'category', 'fiveYearAverageReturn', 'regularMarketPrice', 'logo_url'])

## 1 feb 2021

Need to come of how to add a predicted data to a new column after each prediction.

E.g.

First 60 datasets to predict one value.
Place each of it in a different array.
After predicting that value. Add it to the next array to predict the next value.

## 2 Feb 2021

Fixed issue with predicting data.

Need to recreate the model due to uncomplete data.

## 3 Feb 2021

Formatting and seperating the functions properly and making it look more professional and efficient

## 4 Feb 2021

Create new save feature for models. Created in seperate file model_create.py.

Need to fix the save path for model

Need to test model against real data
