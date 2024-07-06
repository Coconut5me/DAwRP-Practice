#%% - Import Lib
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings("ignore")

#%% - Init data
df = pd.read_csv('data/ACG.csv', index_col="Date", parse_dates=True)
df_close = np.log(df['Close'])
#df_train, val_data= df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]

#%% - Define the number of splits (number of folds) for time series cross-validation
n_splits = 5
max_train_size = len(df)//n_splits
test_size = int((len(df)/n_splits)*0.2)
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

#%% - Iterate over the splits and split the data
for i, (train_index, test_index) in enumerate(tscv.split(df_close)):

    # Split the data based on the indices
    train_data = df_close.iloc[train_index]
    test_data = df_close.iloc[test_index]

    # ARIMA model fitting
    stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
    model = ARIMA(train_data, order=stepwise_fit.order, trend='t')  # Remove the [i] index here
    fitted = model.fit()

    # Forecasting
    fc = fitted.get_forecast(len(test_data))
    fc_values = fc.predicted_mean
    fc_values.index = test_data.index
    conf = fc.conf_int(alpha=0.05)  # 95% confidence interval
    lower_series = conf['lower Close']
    lower_series.index = test_data.index
    upper_series = conf['upper Close']
    upper_series.index = test_data.index

    # Evaluation metrics
    mae = mean_absolute_error(test_data, fc_values)
    mse = mean_squared_error(test_data, fc_values)
    rmse = math.sqrt(mse)

    # Baseline RMSE calculation
    baseline_prediction = np.full_like(test_data, train_data.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test_data, baseline_prediction))

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(df_close, label="Actual data")
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color='orange', label="Test data")
    plt.plot(fc_values, color='red', label="Predict data")
    plt.fill_between(lower_series.index, lower_series, upper_series, color='blue', alpha=.10)
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.title(f"Stock price prediction {i}")
    plt.legend()
    plt.show()

    # Visualize RMSE comparison
    print('ARIMA Model RMSE: {:.2f}'.format(rmse))
    print('Baseline RMSE: {:2f}'.format(baseline_rmse))

    plt.figure(figsize=(16, 10), dpi=150)
    plt.bar(['ARIMA Model', 'Baseline'], [rmse, baseline_rmse], color=['blue', 'green'])
    plt.title('Root Mean Squared Error (RMSE) Comparison')
    plt.ylabel("RMSE")
    plt.show()