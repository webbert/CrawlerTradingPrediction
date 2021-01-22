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

### MODEL USED FOR PREDICTION:

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
