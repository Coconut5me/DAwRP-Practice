#%% - Import Lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import indices
from pmdarima import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import cross_validation
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pickle
from func import *
import warnings
warnings.filterwarnings("ignore")

#%% - Load data
df = pd.read_csv('./data/ACG.csv', index_col="Date", parse_dates=True)
df.info()
df_close = np.log(df['Close'])

#%% - Divide data
n_folds=5
split_train_rate=0.8

#%% - Save model with rolling window
train_list, val_list, df_fold_rolling_window=rolling_window(df_close,n_folds,split_train_rate)
for i in range(len(train_list)):
    model = fit_arima(train_list[i])
    #Save model
    with open(f'time_series/models/rolling_window/arima_rolling_window_{n_folds}_folds_{i}', 'wb') as f:
        pickle.dump(model, f)

#%% - Save model with expanding window
train_list, val_list, df_fold_expanding_window=expanding_window(df_close,n_folds,split_train_rate)
for i in range(len(train_list)):
    model = fit_arima(train_list[i])
    #Save model
    with open(f'time_series/models/expanding_window/arima_expanding_window_{n_folds}_folds_{i}', 'wb') as f:
        pickle.dump(model, f)