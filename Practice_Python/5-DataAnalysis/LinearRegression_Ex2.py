#%% - Import Lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

#%% - Load data
df = pd.read_csv('data/Income.csv')
Income = df["Income"]
Expenditure = df["Expenditure"]
Income, Expenditure = np.array(Income), np.array(Expenditure)
Income = sm.add_constant(Income)

#%% - Create model
model = sm.OLS(Expenditure, Income)
results = model.fit()

#%% - Get results
R_sq = results.rsquared
params = results.params
print(results.summary())

plt.scatter(df["Income"], df["Expenditure"])
plt.plot(df["Income"], params[0] + params[1] * df["Income"])
plt.show()

