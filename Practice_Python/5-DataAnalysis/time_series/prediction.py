 #%% - Import Lib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pickle
from func import *

#%% - Load data
df = pd.read_csv('./data/ACG.csv', index_col="Date", parse_dates=True)
df.info()
df_close = np.log(df['Close'])

#%% - Config dataset
n_folds=5
split_train_rate=0.8

#%% - Divide train - val - test

# - Divide train - test
test_data = df_close[int(len(df_close)*0.9):]

# - Divide train - val
train_list_rolling, val_list_rolling, df_fold_rolling_window=rolling_window(df_close,n_folds,split_train_rate)
train_list_expanding, val_list_expanding, df_fold_expanding_window=expanding_window(df_close,n_folds,split_train_rate)

train_list = {}
val_list = {}

# - For Rolling window
train_list['rolling'] = train_list_rolling
val_list['rolling'] = val_list_rolling

# - For Expanding window
train_list['expanding'] = train_list_expanding
val_list['expanding'] = val_list_expanding

#%% - Create df_evaluation
columns = ['Model', 'MSE_val', 'RMSE_val', 'MAE_val',]
df_evaluation = pd.DataFrame(columns=columns)

#%% - Load model with rolling window
for key in train_list.keys():  # lặp qua các key trong train_list (hoặc test_list, vì chúng có cùng keys)
    for i, (train, val) in enumerate(zip(train_list[key], val_list[key])):
        with open(f'time_series/models/{key}_window/arima_{key}_window_{n_folds}_folds_{i}', 'rb') as f:
            fitted = pickle.load(f)
            # print(fitted.summary())

            # - Forecasting
            fc = fitted.get_forecast(len(val))
            fc_values = fc.predicted_mean
            fc_values.index = val.index

            fc_test = fitted.get_forecast(len(test_data))
            fc_values_test = fc_test.predicted_mean
            fc_values_test.index = test_data.index

            # - Confidence interval
            conf_int = fc.conf_int()

            # - Evaluation metrics
            mae_val = mean_absolute_error(val, fc_values)
            mse_val = mean_squared_error(val, fc_values)
            rmse_val = math.sqrt(mse_val)

            mae_test = mean_absolute_error(test_data, fc_values_test)
            mse_test = mean_squared_error(test_data, fc_values_test)
            rmse_test = math.sqrt(mse_test)

            # - Calculate RMSE for baseline
            baseline_prediction = np.full_like(val, train.mean())  # median
            baseline_rmse = np.sqrt(mean_squared_error(val, baseline_prediction))

            # Append evaluation metrics to the DataFrame
            df_evaluation = df_evaluation._append({'Model': f'ARIMA {key}_{i}',
                                                   'MSE_val': mse_val,
                                                   'RMSE_val': rmse_val,
                                                   'MAE_val': mae_val,
                                                   'MSE_test': mse_test,
                                                   'RMSE_test': rmse_test,
                                                   'MAE_test': mae_test,
                                                   }, ignore_index=True)
            # - Plot actual vs predicted values
            plt.figure(figsize=(16, 10), dpi=150)
            plt.plot(df_close)
            plt.plot(train, label="Train data")
            plt.plot(val.index, val, color='green', label="Validation data")
            plt.plot(fc_values.index, fc_values, color='red', label="Prediction on Validation")
            plt.plot(test_data, color='pink', label="Test data")
            plt.plot(fc_values_test, color='purple', label="Prediction on Test")
            plt.fill_between(fc_values.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='blue', alpha=0.1)
            plt.title(f'Stock price predection with {key} window {n_folds} folds_{i}')
            plt.xlabel("Time")
            plt.ylabel("Stock price")
            plt.legend(loc='upper left')
            plt.show()

            # - Visualize RMSE comparison
            print('ARIMA Model RMSE: {:.2f}'.format(rmse_val))
            print('Baseline RMSE: {:2f}'.format(baseline_rmse))

            plt.figure(figsize=(16, 10), dpi=150)
            plt.bar(['ARIMA Model', 'Baseline'], [rmse_val, baseline_rmse], color=['blue', 'green'])
            plt.title('Root Mean Squared Error (RMSE) Comparison')
            plt.ylabel("RMSE")
            plt.show()
