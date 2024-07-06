#%% - Import Lib
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import indices
from pmdarima import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings("ignore")

#%% - Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

#%% - Load data
df = pd.read_csv('data/ACG.csv', index_col="Date", parse_dates=True) #chuyển cột date đóng vai trò index --> kq mâ
# cột date
df.info()

#%% - Draw chart
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Close prices")
plt.show()

#%% - Divide dataset "train" and "test"
df_close = np.log(df['Close']) #use log to scale data
# train_data, test_data = df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]
train_data, test_data= train_test_split(df_close, test_size=0.1, shuffle=False)
plt.plot(train_data, 'blue', label = 'Train data')
plt.plot(test_data, 'red', label = 'Test data')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

#%% - Phân rã chuỗi dữ liệu
# Biểu đồ lịch sử so sánh gía đóng cửa với giá trị trung bình cà độ lệch chuẩn của 12 k trước đó
rolmean = train_data.rolling(12).mean() #rolling.mean(): tính trung bình cửa sổ trượt
# mục đích của rolling là xem xét chênh lệch hiện tại với 12 kỳ trước
rolstd = train_data.rolling(12).std()
plt.plot(train_data, color='blue', label='Original')
plt.plot(rolmean, color='red', label='Rolling mean')
plt.plot(rolstd, 'black', label='Rolling Std')
plt.legend()
plt.show()

#Phân rã chuỗi thời gian (decompose)
decompose_results = seasonal_decompose(train_data, model="multiplicative", period=30)
decompose_results.plot()
plt.show()

#%% - Kiểm định tính dừng của dữ liệu (Station)
def adf_test(data):
    indices = ["ADF: Test statistic", "p-value", "# of Lags", "# of Observations"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value

    if results[1] <=0.05: # (p-value < 0.05)
        print("Rejected the null hypothesis (H0), \nthe data is stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data is stationary")

    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic", "p-value", "# of Lags"]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical Value ({key})"] = value
    if results[1] <= 0.05:  # (p-value < 0.05)
        print("Rejected the null hypothesis (H0), \nthe data is stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data is stationary")

    return results


print(adf_test(train_data))
print("-----"*5)
print(kpss_test(train_data))

#%% - Kiểm định tự tương quan (Auto Correlation)
pd.plotting.lag_plot(train_data)
plt.show()

#%%
plot_pacf(train_data)
plt.show()

#%%
plot_acf(train_data)
plt.show()

#%% - Chuyển đổi dữ liệu --> chuỗi dừng
diff = train_data.diff(1).dropna()
#Biểu đồ thể hiêện dữ liệu ban đầu và sau khi sai phân
fig, ax= plt.subplots(2, sharex="all")
train_data.plot(ax=ax[0], title="Gía đóng cửa")
diff.plot(ax=ax[1], title="Sai phân bậc nhất")
plt.show()

#%%
diff.plot(kind='box', title='Box Plot of Close')
plt.show()

#%%
plot_acf(diff) # --> xác định tham số "q" cho mô hình ARIMA
plt.show()

plot_pacf(diff) # --> xác định tham số "q" cho mô hình ARIMA
plt.show()


#%% - Xác định tham số p,d,q cho mô hình ARIMA
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%% - Tạo model
model = ARIMA(train_data, order=(1,1,2),)
fitted = model.fit()
print(fitted.summary())

#%% - Dự báo (forecast)
# fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)
# fc_series = pd.Series(fc, index=test_data.index)
# lower_series = pd.Series(conf[:,0], index=test_data.index)
# upper_series = pd.Series(conf[:,1], index=test_data.index)

fc = fitted.get_forecast(len(test_data))
fc_values = fc.predicted_mean
fc_values.index = test_data.index
conf = fc.conf_int(alpha=0.05) #95% conf
lower_series = conf['lower Close']
lower_series.index = test_data.index
upper_series = conf['upper Close']
upper_series.index = test_data

#%% - Đánh gía hiệu suất mô hình
mse = mean_squared_error(test_data, fc_values)
print('Test MSE: %3f' % mse)
rmse = math.sqrt(mse)
print('Test RMSE: %3f' % rmse)

#%% - Calculate RMSE for baseline
baseline_prediction = np.full_like(test_data, train_data.mean()) #median
baseline_rmse = np.sqrt(mean_squared_error(test_data, baseline_prediction))

# Reconstruct the original logged series from the first difference
reconstructed_log_series = np.r_[df_close.iloc[0], diff.cumsum() + df_close.iloc[0]]

# Calculate the exponential to get back to the original scale
reconstructed_original_series = np.exp(reconstructed_log_series)

# Plotting the actual vs predicted in original scale
plt.figure(figsize=(16, 10), dpi=150)
plt.plot(df_close.index, df_close, label="Original Actual Stock Price", color='blue')
plt.plot(reconstructed_original_series.index, reconstructed_original_series, label="Reconstructed Original Stock Price", color='orange')
plt.title("Comparison of Actual and Reconstructed Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
#%% - Visualize RMSE comparison
print('ARIMA Model RMSE: {:.2f}'.format(rmse))
print('Baseline RMSE: {:2f}'.format(baseline_rmse))

plt.figure(figsize=(16,10), dpi=150)
plt.bar(['ARIMA Model', 'Baseline'], [rmse, baseline_rmse], color=['blue','green'])
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.ylabel("RMSE")
plt.show()