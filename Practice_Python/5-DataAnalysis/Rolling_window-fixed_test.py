#%% - Import Lib
import math

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
import warnings
warnings.filterwarnings("ignore")


#%% - Config
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

#%% - Load data
df = pd.read_csv('data/ACG.csv', index_col="Date", parse_dates=True)
df.info()

#%% - Draw data after log
df_close = np.log(df['Close'])
plt.plot(df_close)
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.show()

#%% Define rolling window parameters
n_windows = 5  # Number of rolling windows
window_size = len(df_close) // n_windows  # Size of each window

# Initialize lists to store evaluation metrics
columns = ['Window', 'MSE', 'RMSE', 'MAE', 'MSE_test', 'RMSE_test', 'MAE_test']
df_evaluation = pd.DataFrame(columns=columns)
baseline_rmse_list = []

test_data = df_close[int(len(df_close)*0.9):]

#%% Iterate over each rolling window
for i in range(n_windows):
    start_index = i * window_size  # Start index of the window
    end_index = min((i + 1) * window_size, len(df_close))  # End index of the window
    window_data = df_close[start_index:end_index]  # Data within the window

    # Split the window data into training and testing sets
    train_data, val_data = window_data[:int(len(window_data) * 0.8)], window_data[int(len(window_data) * 0.8):]

    # ARIMA model fitting
    stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
    model = ARIMA(train_data, order=stepwise_fit.order, trend='t')
    fitted = model.fit()

    # Forecasting
    fc = fitted.get_forecast(len(val_data))
    fc_values = fc.predicted_mean
    fc_values.index = val_data.index
    conf = fc.conf_int(alpha=0.05)  # 95% confidence interval
    lower_series = conf['lower Close']
    lower_series.index = val_data.index
    upper_series = conf['upper Close']
    upper_series.index = val_data.index

    fc_test = fitted.get_forecast(len(test_data))
    fc_values_test = fc_test.predicted_mean
    fc_values_test.index = test_data.index  # Index adjustment

    # - Đánh gía hiệu suất mô hình
    mae = mean_absolute_error(val_data, fc_values)
    mse = mean_squared_error(val_data, fc_values)
    rmse = math.sqrt(mse)

    mae_test = mean_absolute_error(test_data, fc_values_test)
    mse_test = mean_squared_error(test_data, fc_values_test)
    rmse_test = math.sqrt(mse_test)

    # - Calculate RMSE for baseline
    baseline_prediction = np.full_like(val_data, train_data.mean())  # median
    baseline_rmse = np.sqrt(mean_squared_error(val_data, baseline_prediction))
    baseline_rmse_list.append(baseline_rmse)


    # Append evaluation metrics to the DataFrame
    df_evaluation = df_evaluation._append({'Window': f'Window {i + 1}',
                                          'MSE': mse,
                                          'RMSE': rmse,
                                          'MAE': mae,
                                           'MSE_test': mse_test,
                                           'RMSE_test': rmse_test,
                                           'MAE_test': mae_test,
                                           }, ignore_index=True)

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(df_close, label="Actual data")
    plt.plot(train_data, label="Train data")
    plt.plot(val_data, color='orange', label="Val data")
    plt.plot(fc_values, color='red', label="Predict data")
    plt.plot(test_data, color='green', label="Test data")
    plt.plot(fc_values_test, color='purple', label="Test prediction")
    plt.fill_between(lower_series.index, lower_series, upper_series, color='blue', alpha=.10)
    plt.title(f"Stock price prediction (Window {i + 1})")
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.legend()
    plt.show()

    # - Visualize RMSE comparison
    print('ARIMA Model RMSE: {:.2f}'.format(rmse))
    print('Baseline RMSE: {:2f}'.format(baseline_rmse))

    plt.figure(figsize=(16, 10), dpi=150)
    plt.bar(['ARIMA Model', 'Baseline'], [rmse, baseline_rmse], color=['blue', 'green'])
    plt.title('Root Mean Squared Error (RMSE) Comparison')
    plt.ylabel("RMSE")
    plt.show()

#%% - Visualize overall RMSE comparison
plt.figure(figsize=(16,10))
plt.bar(['ARIMA Model', 'Baseline'], [np.mean(df_evaluation['RMSE']), np.mean(baseline_rmse_list)], color=['blue','green'])
plt.title('Overall Root Mean Squared Error (RMSE) Comparison')
plt.ylabel("RMSE")
plt.show()

print('ARIMA Model RMSE: {:.2f}'.format(np.mean(df_evaluation['RMSE'])))
print('Baseline RMSE: {:2f}'.format(np.mean(baseline_rmse_list)))