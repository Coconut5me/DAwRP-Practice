#%% - Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pickle
import warnings

from statsmodels.graphics.tukeyplot import results

warnings.filterwarnings("ignore")

#%% - Configs
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100
plt.rcParams[f"font.size"] = 14

#%% - Load data
df = pd.read_csv("data/Income.csv")
Income = df[["Income"]]
Expenditure = df[["Expenditure"]]
Income = np.array(Income)
Expenditure = np.array(Expenditure)
Income = sm.add_constant(Income)

#%% - Create model
model = sm.OLS(Expenditure, Income)
results = model.fit()

#%% - Get results
R_sq = results.rsquared
params = results.params
print(model.summary())




