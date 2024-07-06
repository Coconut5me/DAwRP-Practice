#%% - Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

#%% - Configs
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100
plt.rcParams[f"font.size"] = 14

#%% - Load data
df = pd.read_excel('data/Sales.xlsx')

#%% - 1. Vẽ biểu đồ phân tán của lượng bán theo giá sản phẩm và chi phí quảng cáo
# Biểu đồ phân tán của lượng bán theo giá sản phẩm
plt.figure(figsize=(8, 5))
plt.scatter(df['Price'], df['Sales_Volume'], label='Lượng bán', color='blue')
plt.xlabel('Giá sản phẩm ($)')
plt.ylabel('Lượng bán')
plt.title('Biểu đồ phân tán của lượng bán theo giá sản phẩm')
plt.legend()
plt.grid(True)
plt.show()

# Biểu đồ phân tán của lượng bán theo chi phí quảng cáo
plt.figure(figsize=(8, 5))
plt.scatter(df['Ads_Cost'], df['Sales_Volume'], label='Lượng bán', color='red')
plt.xlabel('Chi phí quảng cáo ($)')
plt.ylabel('Lượng bán')
plt.title('Biểu đồ phân tán của lượng bán theo chi phí quảng cáo')
plt.legend()
plt.grid(True)
plt.show()

#%% - 2. Vẽ biểu đồ nhiệt (heatmap) thể hiện mức tương quan giữa giá sản phẩm, chi phí quảng cáo đối với lượng bán
df1 = df.drop(columns=['Week'])
corr_matrix = df1.corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Biểu đồ nhiệt thể hiện tương quan giữa giá sản phẩm, chi phí quảng cáo và lượng bán', fontsize="12")
plt.show()

#%% - 3. Ước lượng các hệ số của mô hình
X = df[['Price', 'Ads_Cost']]
y = df['Sales_Volume']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Print
print("Hệ số hồi quy ước lượng:")
print(f"a = {model.intercept_}")
print(f"b = {model.coef_[0]}")
print(f"c = {model.coef_[1]}")

#%% - 5. Kiểm định độ phù hợp của mô hình với mức ý nghĩa 5%
# Predictions
y_pred = model.predict(X)

# Residuals
residuals = y - y_pred

# Degrees of freedom
n = len(y)
p = X.shape[1]
df_total = n - 1
df_residual = n - p - 1

# Mean squared error (MSE)
mse = np.mean(residuals ** 2)

# R-squared
sst = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (np.sum(residuals ** 2) / sst)

# F-statistic
f_statistic = (r_squared / p) / ((1 - r_squared) / df_residual)

# p-value
p_value = 1 - stats.f.cdf(f_statistic, p, df_residual)

# Print result
print("f_statistic = ", f_statistic)
print("p_value = ", p_value)

# Check significance
alpha = 0.05
if p_value < alpha:
    print(f"Mô hình có ý nghĩa thống kê với mức ý nghĩa {alpha*100}%")
else:
    print(f"Mô hình không có ý nghĩa thống kê với mức ý nghĩa {alpha*100}%")

#%% - 6. Tiên lượng doanh số khi
new_data = np.array([[4.3, 410], [4.9, 440]])
predictions = model.predict(new_data)

print("Tiên lượng doanh số:")
for i in range(len(predictions)):
    print(f"Với giá sản phẩm là ${new_data[i][0]} và chi phí quảng cáo là ${new_data[i][1]}:")
    print(f"Ước lượng lượng bán: {predictions[i]}")