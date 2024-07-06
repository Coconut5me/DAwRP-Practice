#%% - Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings("ignore")

#%% - Configs
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100
plt.rcParams[f"font.size"] = 14

#%% - Load data
df = pd.read_csv("data/Income.csv")
x = df[["Income"]]
y= df[["Expenditure"]]

#%% - Fit model
model = LinearRegression().fit(x, y)

#%% - Get results
intercept = model.intercept_
slope = model.coef_
R_2 = model.score(x, y)

#%% - Visualization
plt.scatter(x, y)
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.plot(x, intercept + slope * x)
plt.show()


#%% - Prediction
# pre_values = intercept + slope * x
pre_values = model.predict(x)
plt.plot(x, y ,label="Actual")
plt.plot(x, pre_values, label="Predicted")
plt.legend()
plt.show()

#%% -
others_incomes = np.array([35, 42, 28]).reshape(-1,1)
#pre_future_values = model.predict(others_incomes)
pre_future_values = intercept + slope * others_incomes

#%% - Save model
with open('models/linearRegression', 'wb') as f:
    pickle.dump(model, f)

#%% - Load model
with open('models/linearRegression', 'rb') as f:
    model2 = pickle.load(f)
pre_future_values_2 = model2.predict(others_incomes)
